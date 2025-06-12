# my_pytorch_package/utils/hdf5_utils.py

import h5py # type: ignore
import numpy as np
from io import BytesIO
from typing import Any, Dict, List, Optional, Callable
import litdata as ld # type: ignore

from functools import wraps

def create_hdf5_chunk(
    image_array: np.ndarray,
    other_data: Dict[str, np.ndarray],
    compression: Optional[str] = "gzip",
    compression_opts: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Create a data chunk with image data stored channel-wise and other data stored normally.

    Args:
        image_array (np.ndarray): Multi-channel image array of shape (H, W, C).
        other_data (Dict[str, np.ndarray]): Dictionary of other data arrays (e.g., segmentation maps).
        compression (Optional[str]): Compression algorithm. Defaults to 'gzip'.
        compression_opts (Optional[int]): Compression level. Defaults to None.

    Returns:
        Dict[str, Any]: A chunk containing HDF5 data and metadata.
    """
    buffer = BytesIO()
    with h5py.File(buffer, 'w') as hdf5_file:
        # Store image channels separately
        for channel_idx in range(image_array.shape[-1]):
            hdf5_file.create_dataset(
                f'channel_{channel_idx}',
                data=image_array[..., channel_idx],
                compression=compression,
                compression_opts=compression_opts
            )
    buffer.seek(0)
    chunk = {'hdf5_data': buffer.read()}
    if other_data:
        chunk.update(other_data)
    return chunk

def load_hdf5_chunk(
    hdf5_bytes: bytes,
    channels_to_select: Optional[List[int]] = None,
) -> Dict[str, np.ndarray]:
    """
    Load selected channels and other datasets from HDF5 bytes.

    Args:
        hdf5_bytes (bytes): The HDF5 data as bytes.
        channels_to_select (Optional[List[int]]): List of channel indices to load.

    Returns:
        Dict[str, np.ndarray]: Dictionary containing loaded data.
    """
    buffer = BytesIO(hdf5_bytes)
    data_dict = {}
    with h5py.File(buffer, 'r') as hdf5_file:
        # Load selected channels
        if channels_to_select is not None:
            channels = []
            for channel_idx in channels_to_select:
                channel_name = f'channel_{channel_idx}'
                if channel_name in hdf5_file:
                    channels.append(hdf5_file[channel_name][:])
                else:
                    raise KeyError(f"Channel '{channel_name}' not found in HDF5 file.")
            data_dict['image'] = np.stack(channels, axis=0)
    return data_dict


class DecoratedProcessSample:
    """
    A callable class that decorates the user's `process_sample` function
    to handle multi-channel data appropriately.
    """

    def __init__(self, process_sample: Callable):
        self.process_sample = process_sample

    def __call__(self, *args, **kwargs) -> Any:
        # Call the user's process_sample function
        result = self.process_sample(*args, **kwargs)

        if isinstance(result, dict) and "image" in result:
            image = result["image"]
            if image.ndim == 3 and image.shape[-1] > 3:
                # Multi-channel data, call create_chunk_fn
                image_array = result.pop("image")
                other_data = result  # Everything else is `other_data`
                return create_hdf5_chunk(
                    image_array=image_array,
                    other_data=other_data,
                    compression="gzip",
                )
        # For data with <= 3 channels, return as is
        return result

def optimize(
    fn: Callable,
    **kwargs
):
    """
    Override of `litdata.optimize` to automatically decorate the `process_sample` function.

    Args:
        fn: A function to be executed over each input element. The function should return the data sample that
            corresponds to the input. Every invocation of the function should return a similar hierarchy of objects,
            where the object types and list sizes don't change.
        inputs: A sequence of input to be processed by the `fn` function, or a streaming dataloader.
        output_dir: The folder where the processed data should be stored.
        input_dir: Provide the path where your files are stored. If the files are on a remote storage,
            they will be downloaded in the background while processed.
        weights: Provide an associated weight to each input. This is used to balance work among workers.
        chunk_size: The maximum number of elements to hold within a chunk.
        chunk_bytes: The maximum number of bytes to hold within a chunk.
        compression: The compression algorithm to use over the chunks.
        encryption: The encryption algorithm to use over the chunks.
        num_workers: The number of workers to use during processing
        fast_dev_run: Whether to use process only a sub part of the inputs
        num_nodes: When doing remote execution, the number of nodes to use. Only supported on https://lightning.ai/.
        machine: When doing remote execution, the machine to use. Only supported on https://lightning.ai/.
        num_downloaders: The number of downloaders per worker.
        num_uploaders: The numbers of uploaders per worker.
        reader: The reader to use when reading the data. By default, it uses the `BaseReader`.
        reorder_files: By default, reorders the files by file size to distribute work equally among all workers.
            Set this to ``False`` if the order in which samples are processed should be preserved.
        batch_size: Group the inputs into batches of batch_size length.
        mode: The mode to use when writing the data. Accepts either ``append`` or ``overwrite`` or None.
            Defaults to None.
        use_checkpoint: Whether to create checkpoints while processing the data, which can be used to resume the
            processing from the last checkpoint if the process is interrupted. (`Default: False`)
        item_loader: The item loader that will be used during loading in StreamingDataset. Determines
                the format in which the data is stored and optimized for loading.
        start_method: The start method used by python multiprocessing package. Default to spawn unless running
            inside an interactive shell like Ipython.
    """
    # Wrap the process_sample function in the DecoratedProcessSample class
    decorated_process_sample = DecoratedProcessSample(fn)

    # Call the original `litdata.optimize` with the decorated function
    ld.optimize(
        fn=decorated_process_sample,
        **kwargs
    )
