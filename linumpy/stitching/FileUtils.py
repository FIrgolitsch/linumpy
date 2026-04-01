#! /usr/bin/env python

"""Defines various classes to manage the slicer data, subjects and studies."""

# TODO: add save and load to classes

import itertools
import logging
import pickle as pcl
import re
from collections import defaultdict
from pathlib import Path

import networkx
import numpy as np

from linumpy.io import data_io
from linumpy.stitching import topology

logger = logging.getLogger(__name__)


class Subject:
    """Defines new subject (mouse) with given ID

    :param new_id: ID or name for this subject

    """

    subj_id = "None"
    data_dir = "None"
    result_dir = "None"

    def __init__(self, new_id):
        """Subject class constructor"""
        self.subj_id = new_id
        self.bin_files: list[Path] = []
        self.data: list = []

    def __str__(self):
        object_str = (
            "Subject object with attributes :\n"
            + f"  - subj_id : '{self.subj_id}'\n"
            + f"  - datadir : '{self.data_dir}'\n"
            + f"  - result_dir : '{self.result_dir}'\n"
            + "  - acqinfo : "
            + str(self.info)
            + "\n"
            + "  - data : "
            + str(self.data[0])
        )
        return object_str

    def setDataDir(self, data_dir, ext=".bin"):
        """Sets input data directory for this subject (tissue or mouse).

        :param data_dir: (str) valid directory.
        :param ext: (str, optional) extension used for the volumes (default is '.bin')

        :returns: True/False if directory is valid or not.

        """
        if Path(data_dir).is_dir():
            self.data_dir = data_dir
            return True
        else:
            return False

    def getDataDir(self):
        return self.data_dir

    def setResultDir(self, result_dir):
        """Sets output data directory for this subject (tissue or mouse)
        INPUT
            valid directory
        OUTPUT
            True/False if directory is valid or not.
        """
        if Path(result_dir).is_dir():
            self.result_dir = result_dir
            return True
        else:
            return False

    def getResultDir(self):
        return self.result_dir

    def getDatafiles(self):
        return self.bin_files

    def setAcqInfo(self, csv_fname):
        """Add the acquisition information files to the subject.

        :param csv_fname: (str) Valid AcqInfo.csv complete filename.

        """
        self.info = data_io.load_acqinfo_from_csv(csv_fname)
        self.createDataFromAcqInfo()

    def getAcqInfo(self):
        return self.info

    def display(self):
        logger.info(f"Id: {self.subj_id}")
        logger.info(f"Data Dir: {self.data_dir}")
        logger.info(f"Result Dir: {self.result_dir}")

    def checkForVolumes(self):
        nx = self.info["nStepX"]
        ny = self.info["nStepY"]
        nz = self.info["nSlice"]
        isFluo = self.info["acqFluo"]

        nFiles = nx * ny * nz
        if isFluo:
            nFiles *= 2
        fileCount = 0
        for file in Path(self.getDataDir()).iterdir():
            if file.suffix == ".bin":
                fileCount += 1

        flag = nFiles == fileCount
        return flag

    def getVolShape(self):
        nx = int(self.info["nAlinesPerBframe"])
        ny = int(self.info["nBframes"])
        nz = int(self.info["nBPixelZ"])
        return [nx, ny, nz]

    def getSlicerGridShape(self):
        """Access the slicer grid shape.

        :returns: [frameX, frameY, frameZ]

        """
        frameX = int(self.info["nStepX"])
        frameY = int(self.info["nStepY"])
        frameZ = int(self.info["nSlice"])
        return [frameX, frameY, frameZ]

    def addData(self, data, name):
        """Adds a data object to the subject

        :param data: (data object) A valid data object
        :param name: (str) Data name (for dictionnary indexing)

        """
        # TODO: Make sure this data object is saved by pickle.
        # Add the data to the object data dictionnary.
        self.data.append(data)

    def createDataFromAcqInfo(self):
        # This data
        this_data = SlicerData(self.data_dir, self.getSlicerGridShape(), "Original Data")
        this_data.volshape = self.getVolShape()
        self.data.append(this_data)

    def __getstate__(self):
        """To control how this class is dumped by pickle"""
        # List data
        datalist = []
        for thisdata in self.data:
            datalist.append(thisdata)

        # List all other dictionary values (either custom or built-in)
        sbj_members = vars(self)

        return (datalist, sbj_members)

    def __setstate__(self, state):
        """To control how this class is loaded by pickle"""
        datalist, sbj_members = state
        self.data = []

        # Adding subjects in each group
        for this_data in datalist:
            self.data.append(this_data)

        # Adding other members
        for key in sbj_members:
            setattr(self, key, sbj_members[key])

        return self


class Study:
    """Defines new study using a set of subjects
    INPUT
        Study name
    OUTPUT
        None.
    """

    study_id = "None"
    result_dir = "None"

    def __init__(self, new_id):
        self.study_id = new_id
        self.categories = defaultdict(list)

    def setResultDir(self, result_dir):
        """Sets output data directory for this study
        INPUT
            valid directory
        OUTPUT
            True/False if directory is valid or not.
        """
        result_dir = Path(result_dir) / self.study_id
        d = result_dir.parent
        if not d.exists():
            d.mkdir(parents=True)
        self.result_dir = str(result_dir)

    def getResultDir(self):
        """Get output data directory for this study
        INPUT
            None
        OUTPUT
            Str containing the output dir path
        """
        return self.result_dir

    def addSubject(self, subject, category="None"):
        """Adds a subject to the study with a
        INPUT
            valid subject
            (optional) category in which to classify the subject
        OUTPUT
            None
        """

        # Create the subject directory within the category it is assigned to
        self.categories[category].append(subject)
        study_dir = Path(self.result_dir) / category / subject.subj_id
        if not study_dir.exists():
            study_dir.mkdir(parents=True)

        # Inform the subject of where result data should be saved
        subject.setResultDir(study_dir)

    def display(self):
        """Will list the name of the study, the result dir, and then list all
        subjects and their classification in the study.
        OUTPUT
            None
        """
        logger.info(f"Study Id: {self.study_id}")
        logger.info(f"Result Dir: {self.result_dir}")
        logger.info(list(self.categories.items()))

    def __getstate__(self):
        """To control how this class is dumped by pickle"""
        # List categories & category per subject & subjects
        categories = []
        subjectCategory = []
        subjects = []

        for group in self.categories:
            categories.append(group)
            subjectList = self.categories[group]
            for subject in subjectList:
                subjectCategory.append(group)
                subjects.append(subject)

        # List all other dictionary values (either custom or built-in)
        study_members = vars(self)

        return (categories, subjectCategory, subjects, study_members)

    def __setstate__(self, state):
        """To control how this class is loaded by pickle"""
        _categories, subjectCategory, subjects, study_members = state
        nSubjects = len(subjectCategory)

        self.categories = defaultdict(list)

        # Adding subjects in each group
        for subject in range(nSubjects):
            this_group = subjectCategory[subject]
            self.categories[this_group].append(subjects[subject])

        # Adding other members
        for key in study_members:
            setattr(self, key, study_members[key])

        return self


class SlicerData:
    """Slicer data class. It can be used to access raw data or to manage a new dataset.

    :param datadir: (int) Path to the data directory.
    :param gridshape: (tuple) Slicer grid shape.
    :param name: (str, default='data') Name of data set.
    :param prototype: (str, default='volume_x%02.0f_y%02.0f_z%02.0f') File name prototype.
    :param extension: (str, default='.bin') File extension
    :param volshape: (list, default=[512, 512, 120]) Volume shape.
    :param pixelFormat: (str, default='float32') Data format

    """

    def __init__(
        self,
        datadir,
        gridshape=None,
        name="data",
        prototype="volume_x%02.0f_y%02.0f_z%02.0f",
        extension=".bin",
        volshape=None,
        pixelFormat="float32",
        detect_data=False,
    ):
        """Creating a new data object"""
        if volshape is None:
            volshape = [512, 512, 120]
        self.datadir = datadir

        # Try to detect the data information
        if detect_data:
            data_info = dataSniffer(self.datadir)
            self.prototype = data_info["prototype"]
            self.extension = data_info["extension"]
            self.gridshape = data_info["gridshape"]
            self.startIdx = [
                data_info["xrange"][0],
                data_info["yrange"][0],
                data_info["zrange"][0],
            ]
        else:
            self.prototype = prototype
            self.extension = extension
            self.gridshape = gridshape
            self.startIdx = [1, 1, 1]

        self.volshape = volshape
        self.name = name
        self.format = pixelFormat
        self.resolution = [1.0, 1.0, 1.0]

        self.set_gridOrigin("top-left")

    def __str__(self):
        object_str = (
            f"<{self.__class__.__name__}> object with attributes :\n"
            + f"  - name : '{self.name}'\n"
            + f"  - datadir : '{self.datadir}'\n"
            + f"  - prototype : '{self.prototype}'\n"
            + f"  - extension : '{self.extension}'\n"
            + "  - volshape : "
            + str(self.volshape)
            + "\n"
            + "  - gridshape : "
            + str(self.gridshape)
            + "\n"
            + f"  - format : '{self.format}'\n"
            + "  - resolution : "
            + str(self.resolution)
            + "\n"
            + "  - startIdx : "
            + str(self.startIdx)
            + "\n"
        )
        return object_str

    def save(self, filename):
        with Path(filename).open("wb") as f:
            pcl.dump(self, f)

    def checkVolShape(self):
        """Load a volume and get its volume shape. Only works for nii of nii.gz files"""
        if self.extension == ".nii" or self.extension == ".nii.gz":
            vol = self.loadFirstVolume()
            self.volshape = vol.shape
        else:
            logger.info("This method only works for nii and nii.gz files. Keeping the original volshape.")

    def set_gridOrigin(self, origin):
        """To define the mosaic grid origin as either: 'top-right', 'top-left', 'down-right' or 'down-left"""
        valid_origins = ["top-left", "top-right", "bottom-right", "bottom-left"]
        assert origin in valid_origins, f"Unknown origin. Must be one of these: {valid_origins}"
        self.grid_origin = origin
        assert self.gridshape is not None, "gridshape must be set before calling set_gridOrigin with a non-default origin"
        if origin == "top-left":
            gridOrigin = (0, 0, 0)
            direction = (1, 1, 1)
        elif origin == "top-right":
            gridOrigin = (self.gridshape[0] - 1, 0, 0)
            direction = (-1, 1, 1)
        elif origin == "bottom-right":
            gridOrigin = (self.gridshape[0] - 1, self.gridshape[1] - 1, 0)
            direction = (-1, -1, 1)
        elif origin == "bottom-left":
            gridOrigin = (0, self.gridshape[1] - 1, 0)
            direction = (1, -1, 1)
        else:
            raise ValueError(f"Invalid origin: {origin}")

        nx, ny, nz = self.gridshape[:]
        self.gridPosConversionMatrix = np.zeros((nx, ny, nz, 3), dtype=np.uint8)
        for x in range(nx):
            for y in range(ny):
                for z in range(nz):
                    xp = int(direction[0] * (x - gridOrigin[0]))
                    yp = int(direction[1] * (y - gridOrigin[1]))
                    zp = int(direction[2] * (z - gridOrigin[2]))
                    self.gridPosConversionMatrix[x, y, z, :] = [xp, yp, zp]

    def convert_posToGridPos(self, pos):
        return self.gridPosConversionMatrix[pos[0], pos[1], pos[2], :]

    def get_tile_path(self, pos):
        x, y, z = pos
        filename = str(
            Path(self.datadir)
            / (self.prototype % (x + self.startIdx[0], y + self.startIdx[1], z + self.startIdx[2]) + self.extension)
        )
        return filename

    def loadVolume(self, pos):
        """Loads a volume from the dataset.

        :param pos: (list) Volume position (in grid reference) to load.
        :returns: ndarray containing the loaded volume.

        Notes
        -----
        - If the volume does'nt exist, returns an arrays of zero with the predefined shape.

        """
        try:
            filename = self.get_tile_path(pos)
            return data_io.load_volumeByFilename(filename, tuple(self.volshape), self.format)
        except Exception:
            return None

    def loadFirstVolume(self):
        """Loads the first non-empty volume"""
        for vol in self.volumeIterator():
            if vol is not None:
                return vol
                break

    def saveVolume(self, vol, pos, overwrite=False):
        """Saves a volume into the dataset directory.

        :param vol: (ndarray) Volume to save
        :param pos: (list) Volume position (in grid reference)
        :param overwrite: (bool, default=False) If set to true, the saved volume will overwrite any pre-existing file.

        .. note:: Only nifti files (*.nii* and *.nii.gz*) can be saved in this version.

        """
        x, y, z = pos
        filename = str(
            Path(self.datadir)
            / (self.prototype % (x + self.startIdx[0], y + self.startIdx[1], z + self.startIdx[2]) + self.extension)
        )

        # Check if datadir exits
        if not Path(self.datadir).exists():
            Path(self.datadir).mkdir(parents=True)

        # Check if file exists
        if not Path(filename).exists() or overwrite:
            if self.extension in [".nii", ".nii.gz"]:
                data_io.save_nifti(filename, vol, pixelFormat=self.format)
            else:
                logger.info(f"Volume save is not implemented yet for extension '{self.extension}'")
                raise NotImplementedError
        else:
            logger.info(f"This file already exists : '{filename}'")

    def volumeIterator(self, returnPos=False, mask=None, returnPosOnly=False):
        """Iterates over all volumes

        :param returnPos: (bool, default=False) If set to True, the iterator will yield
            the position in addition to the volume at each iteration.
        :param mask: (ndarray, default=None) This mask specify which volumes to keep in the iteration.

        :returns: vol
        :returns: vol, pos (if returnPos=True)

        """
        assert self.gridshape is not None
        for z in range(self.gridshape[2]):
            if returnPosOnly:
                for pos in self.sliceIterator(z, returnPos, mask, returnPosOnly):
                    yield pos
            else:
                if returnPos:
                    for vol, pos in self.sliceIterator(z, returnPos, mask):
                        yield vol, pos
                else:
                    for vol in self.sliceIterator(z, returnPos, mask):
                        yield vol

    def sliceIterator(self, z, returnPos=False, mask=None, returnPosOnly=False):
        """Iterates over all volumes in slice z

        :param z: (int) Slice number over which the iteration occurs.
        :param returnPos: (bool, default=False) If set to True, the iterator will yield
            the position in addition to the volume at each iteration.
        :param mask: (ndarray, default=None) This mask specify which volumes to keep in the iteration.

        :returns: vol
        :returns: vol, pos (if returnPos=True)

        """
        assert self.gridshape is not None
        nx = list(range(self.gridshape[0]))
        ny = list(range(self.gridshape[1]))
        for x, y in itertools.product(nx, ny):
            # Ignoring volumes if mask given
            if mask is not None:
                if mask.ndim == 2:
                    if not (mask[x, y]):
                        continue
                else:
                    if not (mask[x, y, z]):
                        continue

            if returnPosOnly:
                yield (x, y, z)
            else:
                vol = self.loadVolume((x, y, z))
                if vol is not None:
                    if returnPos:
                        pos = (x, y, z)
                        yield vol, pos
                    else:
                        yield vol

    def neighborIterator(self, returnPos=False, mask=None, returnPosOnly=False):
        """Iterates over all neighbors

        :param returnPos: (bool, default=False) If set to True, the iterator will yield
            the position in addition to the volume at each iteration.

        :returns: vol1, vol2
        :returns: vol1, vol2, pos1, pos2 (if returnPos=True)

        """
        assert self.gridshape is not None
        # Loop over all slices
        for z in range(self.gridshape[2]):
            if returnPosOnly:
                for pos1, pos2 in self.neighborSliceIterator(z, returnPos, mask, returnPosOnly):
                    yield pos1, pos2
            else:
                if returnPos:
                    for vol1, vol2, pos1, pos2 in self.neighborSliceIterator(z, returnPos, mask=mask):
                        yield vol1, vol2, pos1, pos2
                else:
                    for vol1, vol2 in self.neighborSliceIterator(z, returnPos, mask=mask):
                        yield vol1, vol2

    def neighborSliceIterator(self, z, returnPos=False, mask=None, returnPosOnly=False):
        """Iterates over all neighbors in slice z

        :param returnPos: (bool, default=False) If set to True, the iterator will yield
            the position in addition to the volume at each iteration.
        :param z: (int) Slice number over which the iteration occurs.

        :returns: vol1, vol2
        :returns: vol1, vol2, pos1, pos2 (if returnPos=True)

        """
        assert self.gridshape is not None
        nX = self.gridshape[0]
        nY = self.gridshape[1]
        this_topo = topology.generate_default(nX, nY)

        if mask is not None:
            if mask.ndim == 3:
                mask = mask[:, :, z]
            topology.remove_agarose(this_topo, mask)

        xx = networkx.get_node_attributes(this_topo, "x")
        yy = networkx.get_node_attributes(this_topo, "y")

        # Loop over all edges
        if float(networkx.__version__) < 2.0:
            for u, v in this_topo.edges_iter():
                pos1 = (xx[u], yy[u], z)
                pos2 = (xx[v], yy[v], z)
                if returnPosOnly:
                    yield pos1, pos2
                else:
                    vol1 = self.loadVolume(pos1)
                    vol2 = self.loadVolume(pos2)
                    if vol1 is not None and vol2 is not None:
                        if returnPos:
                            yield vol1, vol2, pos1, pos2
                        else:
                            yield vol1, vol2
        else:
            for u, v in this_topo.edges():
                pos1 = (xx[u], yy[u], z)
                pos2 = (xx[v], yy[v], z)
                if returnPosOnly:
                    yield pos1, pos2
                else:
                    vol1 = self.loadVolume(pos1)
                    vol2 = self.loadVolume(pos2)
                    if vol1 is not None and vol2 is not None:
                        if returnPos:
                            yield vol1, vol2, pos1, pos2
                        else:
                            yield vol1, vol2

    def singlePassNeighborIterator(self, origin, method="bfs", mask=None, returnPosOnly=False):
        """Iterator that traverse the whole dataset in a single pass.

        :param origin: (2x1 array) (grid coordinates (begins at 1))
        :param method: (str, default='bfs') Graph traversing method ('dfs' or 'bfs')

        :returns: vol1, vol2, pos1, pos2

        """
        assert self.gridshape is not None
        for z in range(self.gridshape[2]):
            if returnPosOnly:
                for pos1, pos2 in self.singlePassNeighborSliceIterator(origin, z, method, mask, returnPosOnly):
                    yield pos1, pos2

            else:
                for vol1, vol2, pos1, pos2 in self.singlePassNeighborSliceIterator(origin, z, method, mask):
                    yield vol1, vol2, pos1, pos2

    def singlePassNeighborSliceIterator(self, origin, z, method="bfs", mask=None, returnPosOnly=False):
        """Iterator that traverse slice z in a single pass.

        :param origin: (2x1 array) (grid coordinates (begins at 1))
        :param z: (int) slice number
        :param method: (str, default='bfs') Graph traversing method ('dfs' or 'bfs')

        :returns: vol1, vol2, pos1, pos2

        """
        assert self.gridshape is not None
        topo = topology.generate_default(self.gridshape[0], self.gridshape[1])

        # Remove agarose from topology
        if mask is not None:
            if mask.ndim == 3:
                mask = mask[:, :, z]
            topology.remove_agarose(topo, mask)
        sList, tList = topology.topoIterator(topo, root=origin, method=method)

        # Loop over source and target list
        for source, target in zip(sList, tList, strict=False):
            pos1 = (source[0], source[1], z)
            pos2 = (target[0], target[1], z)
            if returnPosOnly:
                yield pos1, pos2
            else:
                vol1 = self.loadVolume(pos1)
                vol2 = self.loadVolume(pos2)
                if vol1 is not None and vol2 is not None:
                    yield vol1, vol2, pos1, pos2

    def update_gridshape(self):
        self.gridshape = detect_gridshape(self.datadir, self.prototype, self.extension)


def detect_gridshape(datadir, prototype="volume_x%02.0f_y%02.0f_z%02.0f", extension=".bin"):
    # List all files in datadir
    if isinstance(datadir, str):
        fileList = [f.name for f in Path(datadir).iterdir()]
    elif isinstance(datadir, list):
        fileList = datadir
    else:
        raise TypeError(f"datadir must be str or list, got {type(datadir)}")

    # Create a regex expression to find all files matching prototypes
    filename_rx_prototype = prototype + extension

    # Replacing %ds
    filename_rx_prototype = re.sub("%d", r"(?P<x>\d+)", filename_rx_prototype, count=1)
    filename_rx_prototype = re.sub("%d", r"(?P<y>\d+)", filename_rx_prototype, count=1)
    filename_rx_prototype = re.sub("%d", r"(?P<z>\d+)", filename_rx_prototype, count=1)

    # Replacing %fs
    filename_rx_prototype = re.sub("%[0-9]*[.]*[0-9]*f", r"(?P<x>\d+)", filename_rx_prototype, count=1)
    filename_rx_prototype = re.sub("%[0-9]*[.]*[0-9]*f", r"(?P<y>\d+)", filename_rx_prototype, count=1)
    filename_rx_prototype = re.sub("%[0-9]*[.]*[0-9]*f", r"(?P<z>\d+)", filename_rx_prototype, count=1)

    # Prepare sniffer
    filename_rx = re.compile(filename_rx_prototype)
    maxX = 0
    maxY = 0
    maxZ = 0
    minX = None
    minY = None
    minZ = None

    # Loop over all files in directory
    for elem in fileList:
        b = filename_rx.match(elem)
        if b is not None:
            if maxX < int(b.group("x")):
                maxX = int(b.group("x"))
            if maxY < int(b.group("y")):
                maxY = int(b.group("y"))
            if maxZ < int(b.group("z")):
                maxZ = int(b.group("z"))
            if minX is None or minX > int(b.group("x")):
                minX = int(b.group("x"))
            if minY is None or minY > int(b.group("y")):
                minY = int(b.group("y"))
            if minZ is None or minZ > int(b.group("z")):
                minZ = int(b.group("z"))
    try:
        assert minX is not None and minY is not None and minZ is not None
        gridshape = (
            int(maxX) - int(minX) + 1,
            int(maxY) - int(minY) + 1,
            int(maxZ) - int(minZ) + 1,
        )
    except Exception:
        logger.info("Not able to detect gridshape. Setting to 0")
        gridshape = (0, 0, 0)

    return gridshape


def dataSniffer(datadir: str) -> dict:
    """Detect the mosaic information.
    Parameters
    ----------
    datadir: str
        Path to the directory containing the raw data
    Returns
    -------
    data_info: dict
        Dictionary with extracted information.
    """
    filelist = [f.name for f in Path(datadir).iterdir()]
    filename_rx = re.compile(
        r"(?P<prefix>[A-Za-z-_]+)(?P<x>\d+)(?P<bXY>[A-Za-z-_]+)(?P<y>\d+)(?P<bYZ>[A-Za-z-_]+)(?P<z>\d+)(?P<suffix>.*)(?P<ext>\..*)"
    )
    filename_rx_woExt = re.compile(
        r"(?P<prefix>[A-Za-z-_]+)(?P<x>\d+)(?P<bXY>[A-Za-z-_]+)(?P<y>\d+)(?P<bYZ>[A-Za-z-_]+)(?P<z>\d+)(?P<suffix>.*)"
    )

    # Grap all volume-like files
    dataList = []
    prefix = set()
    suffix = set()
    extension = set()
    bXY = set()
    bYZ = set()
    lengthPos = set()
    maxX = 0
    maxY = 0
    maxZ = 0
    minX = None
    minY = None
    minZ = None
    this_extension = ""

    for elem in filelist:
        b = filename_rx.match(elem)
        if b is not None:
            dataList.append(elem)

            # Detect extension
            this_extension = "".join(Path(elem).suffixes)
            filename_wo_ext = elem.replace(this_extension, "")

            # Process the filename again
            b2 = filename_rx_woExt.match(filename_wo_ext)
            if b2 is None:
                continue

            if maxX < int(b2.group("x")):
                maxX = int(b2.group("x"))
            if maxY < int(b2.group("y")):
                maxY = int(b2.group("y"))
            if maxZ < int(b2.group("z")):
                maxZ = int(b2.group("z"))
            if minX is None or minX > int(b2.group("x")):
                minX = int(b2.group("x"))
            if minY is None or minY > int(b2.group("y")):
                minY = int(b2.group("y"))
            if minZ is None or minZ > int(b2.group("z")):
                minZ = int(b2.group("z"))

            prefix.add(b2.group("prefix"))
            suffix.add(b2.group("suffix"))
            # extension.add(b.group("ext"))
            bXY.add(b2.group("bXY"))
            bYZ.add(b2.group("bYZ"))
            lengthPos.add(len(b2.group("x")))
            lengthPos.add(len(b2.group("y")))
            lengthPos.add(len(b2.group("z")))

    assert minX is not None and minY is not None and minZ is not None
    gridshape = (maxX - minX + 1, maxY - minY + 1, maxZ - minZ + 1)

    # Detect extension
    extension = this_extension

    logger.info(f"Xrange: {(minX, maxX)}")
    logger.info(f"Yrange: {(minY, maxY)}")
    logger.info(f"Zrange: {(minZ, maxZ)}")
    logger.info(f"Detected grid shape: {gridshape}")
    logger.info(f"Detected extensions: {extension}")

    # Creating a file prototype
    idxFormat = "%d"
    if min(lengthPos) >= 2:
        idxFormat = f"%0{min(lengthPos)}.0f"
    prototype = next(iter(prefix)) + idxFormat + next(iter(bXY)) + idxFormat + next(iter(bYZ)) + idxFormat + next(iter(suffix))
    logger.info(f"Generated file prototype: {prototype}")

    # Detect missing files
    data_mask = np.zeros(gridshape, dtype=bool)
    assert minX is not None and minY is not None and minZ is not None
    for x in range(minX, maxX + 1):
        for y in range(minY, maxY + 1):
            for z in range(minZ, maxZ + 1):
                filename = prototype % (x, y, z) + extension
                if filename in filelist:
                    data_mask[x - minX, y - minY, z - minZ] = True

    nVols = gridshape[0] * gridshape[1] * gridshape[2]
    logger.info(f"There are {nVols - data_mask.sum()}/{nVols} missing files in this grid.")

    # Creating the output dict
    data_info = {}
    data_info["datadir"] = datadir
    data_info["prototype"] = prototype
    data_info["extension"] = extension
    data_info["gridshape"] = gridshape
    data_info["xrange"] = (minX, maxX)
    data_info["yrange"] = (minY, maxY)
    data_info["zrange"] = (minZ, maxZ)
    data_info["startIdx"] = (minX, minY, minZ)
    data_info["data_mask"] = data_mask

    return data_info
