import os
from struct import unpack

import numpy as np


def read_dpt(dpt_file_path):
    """read depth map from *.dpt file.

    :param dpt_file_path: the dpt file path
    :type dpt_file_path: str
    :return: depth map data
    :rtype: numpy
    """
    TAG_FLOAT = 202021.25  # check for this when READING the file

    ext = os.path.splitext(dpt_file_path)[1]

    assert len(ext) > 0, "readFlowFile: extension required in fname %s" % dpt_file_path
    assert ext == ".dpt", exit(
        "readFlowFile: fname %s should have extension " ".flo" "" % dpt_file_path
    )

    fid = None
    try:
        fid = open(dpt_file_path, "rb")
    except IOError:
        print("readFlowFile: could not open %s", dpt_file_path)

    tag = unpack("f", fid.read(4))[0]
    width = unpack("i", fid.read(4))[0]
    height = unpack("i", fid.read(4))[0]

    assert tag == TAG_FLOAT, (
        "readFlowFile(%s): wrong tag (possibly due to big-endian machine?)"
        % dpt_file_path
    )
    assert 0 < width and width < 100000, "readFlowFile(%s): illegal width %d" % (
        dpt_file_path,
        width,
    )
    assert 0 < height and height < 100000, "readFlowFile(%s): illegal height %d" % (
        dpt_file_path,
        height,
    )

    # arrange into matrix form
    depth_data = np.fromfile(fid, np.float32)
    depth_data = depth_data.reshape(height, width)

    fid.close()

    return depth_data

def disparity2depth(disparity_map,  baseline=1.0, focal=1.0):
    """Convert disparity value to depth value.
    """
    no_zeros_index = np.where(disparity_map != 0)
    depth_map = np.full(disparity_map.shape, np.Inf, np.float64)
    depth_map[no_zeros_index] = (baseline * focal) / disparity_map[no_zeros_index]
    return depth_map


