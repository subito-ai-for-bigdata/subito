import math
import random
from abc import abstractmethod, ABC
import heapq
import numpy as np
import dask.array as da

count = -1

def generate_synopsis_id():
    global count
    count += 1
    return count


class Synopsis(ABC):

    def __init__(self, key_index=None, value_index=None):
        self.synopsis_id = generate_synopsis_id()
        self.key_index = None
        self.value_index = None

    def get_synopsis_id(self):
        return self.synopsis_id

    def set_synopsis_id(self, synopsis_id):
        self.synopsis_id = synopsis_id

    def set_key_index(self, key_index):
        self.key_index = key_index

    def get_key_index(self):
        return self.key_index

    def set_value_index(self, value_index):
        self.value_index = value_index

    def get_value_index(self):
        return self.value_index

    @abstractmethod
    def add(self, key_index, value_index):
        pass

    @abstractmethod
    def estimate(self, key_index):
        pass

    @abstractmethod
    def merge(self, other_synopsis):
        pass

    def operation_mode_add(self, obj):
        # This seems to be specific to the image, and its purpose is unclear.
        # You'll need to provide more context or its intended behavior.
        raise NotImplementedError

    def get_hash_count(self):
        return self.hash_count


class WeightedPrioritySampler(Synopsis):
    """
    A streaming sampler that keeps up to k items with the smallest
    'priority', where priority = -log(U) / weight.
    """

    def __init__(self, k, seed=None):
        self.k = k
        self.heap = []  # will store tuples: (neg_priority, value, weight)
        self.random_state = random.Random(seed)

    def add_serial(self, value, weight=1.0):
        if weight <= 0:
            return
        u = self.random_state.random()
        if u <= 0:
            return
        priority = -math.log(u) / weight
        neg_priority = -priority

        if len(self.heap) < self.k:
            heapq.heappush(self.heap, (neg_priority, value, weight))
        else:
            if neg_priority > self.heap[0][0]:
                heapq.heapreplace(self.heap, (neg_priority, value, weight))

    def add(self, chunk):
        """
        Add chunks of items (values and weights) to the sampler.
        Arguments:
            values: List or array of values.
            weights: List or array of weights (same size as values).
        """

        chunk_shape = chunk.shape
        print(f"chunk_shape: {chunk_shape}")

        # Separate values and weights from the data
        values, weights = np.array(chunk).T

        if np.any(weights <= 0):
            print("negative weights found !!!")
            return

        # Generate random U values
        u_values = np.random.random(size=values.shape)

        if np.any(u_values <= 0):
            print("negative u_values found !!!")
            return

        # Compute priorities
        priorities = -np.log(u_values) / weights
        neg_priorities = -priorities

        # Combine into a single array: (neg_priority, value, weight)
        combined = np.stack([neg_priorities, values, weights], axis=1)

        # Sort combined array by neg_priority (column 0) and select top k
        top_k_indices = np.argsort(-combined[:, 0])[:self.k]  # Sort by neg_priority desc
        print(f"top_k_indices: {top_k_indices}")

        top_k = combined[top_k_indices]  # Get top k rows
        print(f"top k: {top_k.shape}")

        padding = da.zeros((chunk_shape[0], top_k.shape[1]), chunks="auto")

        padding[0:top_k.shape[0], :] = top_k[0:top_k.shape[0], :]

        print(f"padding: {padding}")

        self.heap = top_k

        return top_k

    """
    def merge(self, other):
        for neg_priority, value, weight in other.heap:
            if len(self.heap) < self.k:
               heapq.heappush(self.heap, (neg_priority, value, weight))
            else:
               if neg_priority > self.heap[0][0]:
                  heapq.heapreplace(self.heap, (neg_priority, value, weight))
    """

    def merge(self, tables):
        """
        Merge multiple tables, keeping only the top k rows based on the first column (priority).

        Arguments:
            tables: List of 2D NumPy arrays to merge.
            k: Number of top rows to keep based on the first column (priority).

        Returns:
            2D NumPy array with the top k rows.
        """
        # Concatenate all tables
        # Sort by the first column (priority) in descending order
        merged_table = da.vstack(tables)
        merged_table_top_k = merged_table[da.argtopk(-merged_table[:, 0], self.k)[::-1]]
        # print(f"merged_table_top_k: {merged_table_top_k}")

        self.heap = merged_table_top_k

        return self

    """
    def estimate(self):
        total_weight = 0.0
        weighted_sum = 0.0
        for neg_priority, value, weight in self.heap:
            total_weight += weight
            weighted_sum += value * weight
        if total_weight == 0.0:
            return 0.0
        return weighted_sum / total_weight
    """

    def estimate(self):
        total_weight = da.sum(self.heap[:, 2]).compute()
        # print(f"weights: {self.heap[:, 2]}")
        # print(f"values: {self.heap[:, 1]}")
        values_weights = self.heap[:, 1] * self.heap[:, 2]
        # print(f"values * weights: {values_weights}")
        weighted_sum = da.sum(values_weights).compute()
        if total_weight == 0.0:
            return 0.0
        return weighted_sum / total_weight

    @staticmethod
    def build_sampler_for_partition(partition_data, k, seed):
        """
        Given an iterable of (value, weight) pairs in this partition,
        build and return a WeightedPrioritySampler of size k.
        """
        sampler = WeightedPrioritySampler(k, seed=seed)
        for (value, weight) in partition_data:
            sampler.add(value, weight)
        return sampler


class PrioritySampler(WeightedPrioritySampler):
    """
    A streaming sampler that keeps up to k items with the smallest
    random priority (uniform sampling).

    We store (neg_priority, item) in a min-heap (by neg_priority),
    which simulates a max-heap for actual priority.
    """

    def __init__(self, k, seed=None):
        """
        Parameters
        ----------
        k : int
            Maximum number of items to sample.
        seed : int or None, optional
            Random seed for reproducible priorities.
        """
        self.k = k
        # self.heap = []  # will store (neg_priority, item)
        self.random_state = random.Random(seed)

    def add(self, item):
        """
        Add a single item with a random priority.
        """
        # Generate a priority in [0, 1).
        priority = self.random_state.random()
        # We store negative priority, so the item with the largest
        # actual priority is on top of the min-heap (heap[0]).
        neg_priority = -priority

        if len(self.heap) < self.k:
            # If the heap is not full, just push the new item.
            heapq.heappush(self.heap, (neg_priority, item))
        else:
            # If the heap is full, compare with the top item.
            # If the new item has a smaller priority => bigger neg_priority => replace
            if neg_priority > self.heap[0][0]:
                heapq.heapreplace(self.heap, (neg_priority, item))

    def add_chunk(self, chunk):
        """
        Add chunks of items to the sampler.
        """
        # np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

        chunk = chunk[:, 0]
        chunk_shape = chunk.shape

        #print(f"chunk: {chunk}")

        #if np.any(chunk <= 0):
        #print("negative chunk found !!!")
        #return

        priorities = da.random.random(size=chunk_shape)

        neg_priorities = -priorities

        # Combine into a single array: (neg_priority, value, weight)
        combined = da.stack([neg_priorities, chunk], axis=1)
        # print(f"combined: {combined.shape}")
        # Sort combined array by neg_priority (column 0) and select top k
        top_k_indices = da.argtopk(combined[:, 0], self.k)  # Sort by neg_priority desc
        #print(f"top_k_indices: {top_k_indices}")
        #print(f"combined: {combined.shape}")
        # top_k = combined[top_k_indices]  # Get top k rows
        # #print(f"top_k_shape: {top_k.shape[0]} & {top_k.shape[1]}")
        # #padding = da.zeros((chunk_shape[0], top_k.shape[1]), chunks = "auto")
        #
        # #padding[0:top_k.shape[0], :] = top_k[0:top_k.shape[0], :]
        #
        # self.heap = top_k
        #print(f"top k: {top_k.shape}")
        #print(f"padding: {padding}")

        return top_k_indices

    def reservoir_priority_sampling(self, all_data, k, divided_by):
        num_elements = len(all_data)
        self.k = int(k) // divided_by
        zeros = da.zeros((num_elements), chunks=(num_elements // divided_by))

        # Apply rechunk to control the chunk size
        all_data_indexes = da.arange(num_elements, chunks=(num_elements // divided_by))
        data = da.stack([all_data_indexes, zeros], axis=1)
        data = data.rechunk(num_elements // divided_by, 1)  # Control the chunk size here
        top_ks_splitted = data.map_blocks(self.add_chunk, dtype=all_data.dtype)
        top_ks_splitted = top_ks_splitted.compute()
        # Print the overall length

        array_version = np.array(top_ks_splitted).flatten()
        # merged_PS = self.merge(top_ks_splitted)
        # sampled_indexes = self.estimate()
        return all_data[array_version].compute()

    """
    def merge(self, other):
        #Merge another PrioritySampler into this one, preserving only
        #the top k smallest priorities overall.

        for neg_priority, item in other.heap:
            if len(self.heap) < self.k:
                heapq.heappush(self.heap, (neg_priority, item))
            else:
                if neg_priority > self.heap[0][0]:
                    heapq.heapreplace(self.heap, (neg_priority, item))
    """

    def estimate(self):
        # if not self.heap:
        #     return da.array([])  # Return an empty Dask array if heap is empty
        #
        # heap_array = da.array(self.heap)  # Convert list to Dask array
        # return_array = da.round(heap_array[:, 1]).astype(int)
        # return return_array
        # print(f"values: {da.array(self.heap[:, 1]).compute()}")
        return_array = da.round(self.heap[:, 1]).astype(int)
        return return_array

    def get_sample(self):
        """
        Return the sampled items (in no particular order).
        """
        return [item for (neg_priority, item) in self.heap]

    @staticmethod
    def build_sampler_for_partition(partition_data, k, seed):
        """
        Create a PrioritySampler for this partition's data.
        partition_data is an iterable of items (all unweighted).
        """
        sampler = PrioritySampler(k=k, seed=seed)
        for item in partition_data:
            sampler.add(item)
        return sampler



class PartialDFTAccumulator:
    def __init__(self, dft_sum=None):
        self.dft_sum = dft_sum  # Holds the summed DFT (complex array)
        self.n = None  # Length of each partial signal

    def add(self, chunk):
        """
        Compute the DFT of this chunk (1D array) and store it (by summation).
        """
        if self.n is None or self.n == 0:
            self.n = len(chunk)
        elif len(chunk) != self.n:
            raise ValueError("All chunks must have the same length to sum DFTs.")

        partial_dft = np.fft.fft(chunk)

        if self.dft_sum is None:
            self.dft_sum = partial_dft
        else:
            self.dft_sum += partial_dft
        return self

    def merge(self, *others):
        """
        Merge multiple PartialDFTAccumulators by summing their DFTs.
        """
        for other in others:
            if other.dft_sum is None:
                continue  # Skip empty accumulators

        if self.dft_sum is None:
            self.dft_sum = others[0].dft_sum
            self.n = others[0].n
        else:
            other_dfts = [other.dft_sum for other in others if other.dft_sum is not None]
            self.dft_sum += np.sum(other_dfts, axis=0)

        return self

    def estimate(self):
        """
        Return the final merged DFT.
        """
        if self.dft_sum is None:
            return None  # no data
        return self.dft_sum

    @staticmethod
    def build_accumulator_for_partition(chunk):
        """
        Convert a chunk into a PartialDFTAccumulator.
        """
        acc = PartialDFTAccumulator()
        acc.add(chunk)
        return acc

    @staticmethod
    def merge_many_accumulators(*accumulators):
        """
        Merge multiple PartialDFTAccumulators into one.
        """
        return accumulators[0].merge(*accumulators[1:])


def merge_samplers(sampler_a, sampler_b):
    """
    Merge sampler_b into sampler_a and return sampler_a.
    """
    sampler_a.merge(sampler_b)
    return sampler_a
