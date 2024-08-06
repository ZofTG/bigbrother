"""devices module"""

#! IMPORTS


import ctypes as ct
import platform
import sys
from abc import ABC, abstractmethod
from datetime import datetime
from os import makedirs, remove, getcwd
from os.path import join, dirname
from threading import Thread
from typing import Any, Callable, Tuple, Union

import clr
import cv2
import numpy as np
from numpy.typing import NDArray
from PyQt5 import QtMultimedia

__all__ = [
    "TIMESTAMP_FORMAT",
    "Device",
    "Signal",
    "LeptonDevice",
    "OptrisPiDevice",
    "OpticalDevice",
]


#! CONSTANTS


TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S.%f"
root = dirname(dirname(__file__))
LEPTON_PATH = join(root, "assets", "flir", "lepton")
PI_PATH = join(root, "assets", "optris", "pi")


#! CLASSES


class Signal:
    """
    class allowing to generate event signals and connect them to functions.
    """

    _connected_fun: Callable

    def __init__(self):
        self._connected_fun = self._none_fun

    @property
    def connected_function(self):
        """return the function connected to this signal."""
        return self._connected_fun

    def _none_fun(self, *args, **kwargs):
        """private method used to supply the lack of provided callbacks"""
        return None

    def emit(self, *args, **kwargs):
        """emit the signal with the provided parameters."""
        if self.is_connected:
            return self.connected_function(*args, **kwargs)
        return None

    def connect(self, fun: Callable):
        """
        connect a function/method to the actual signal

        Parameters
        ----------
        fun: FunctionType | MethodType
            the function to be connected to the signal.
        """
        assert isinstance(fun, Callable), "fun must be a Callable object."
        self._connected_fun = fun

    def disconnect(self):
        """disconnect the signal from the actual function."""
        self._connected_fun = self._none_fun

    def is_connected(self):
        """check whether the signal is connected to a function."""
        return (self._connected_fun is not None) and (
            isinstance(self._connected_fun, Callable)
        )


class Device(ABC):
    """
    Abstract class that allows to deal with different image capturing devices
    """

    _id: str
    _streaming: bool
    _fps: Union[float, None]
    _rot_angle: float
    _last: Union[None, Tuple[datetime, NDArray[Union[np.uint8, np.float_]]]]
    _last_changed: Signal
    _connected: bool

    @staticmethod
    def get_available_devices():
        """
        return the list of camera devices.

        Returns
        -------
        devices: List[str]
            a list of connected cameras.
        """
        opt_dev = QtMultimedia.QCameraInfo.availableCameras()
        idx = np.argsort([i.deviceName() for i in opt_dev])
        return np.array([i.description() for i in opt_dev])[idx].tolist()

    @property
    def fps(self):
        """return the actual fps"""
        return self._fps

    @property
    def rot_angle(self):
        """return the camera rotation angle"""
        return self._rot_angle

    @property
    def last_sample(self):
        """return last obtained frame"""
        return self._last

    @property
    def last_changed(self):
        """return the signal connected to the change of the last sample"""
        return self._last_changed

    @property
    def id(self):
        """returns the device's ID (name + serial number)."""
        return self._id

    @property
    def streaming(self):
        """returns True if the device is streaming data."""
        return self._streaming

    @property
    def connected(self):
        """return the connection status"""
        return self._connected

    def set_rotation_angle(self, angle: Union[float, int]):
        """
        set the required rotation angle in degrees.

        Parameters
        ----------
        angle: Union[float, int]
            the required rotation angle in degrees
        """
        if not isinstance(angle, (float, int)):
            raise ValueError
        self._rot_angle = angle

    def set_last_sample(self, image: NDArray[Any]):
        """
        update the last sample data and trigger the appropriate update event

        Parameters
        ----------
        image : NDArray[Any]
            the 3D array containing the image data
        """

        # get the data
        img = self.rotate_image(image)
        dt = datetime.now()

        # update the last sample and the true fps
        if self._last is None:
            lapse_time = 0
            self._fps = None
        else:
            lapse_time = (dt - self._last[0]).total_seconds()
            if lapse_time != 0:
                self._fps = lapse_time**-1
            else:
                self._fps = None
        self._last = (dt, img)

        # emit the last_sample changed signal
        self._last_changed.emit()

    def rotate_image(self, image: NDArray[Any]):
        """
        rotate an image around its center

        Parameters
        ----------
        image: NDArray[Any]
            the 3D array to be rotated

        Returns
        -------
        rotated: NDArray[Any]
            the 3D array rotated by the imposed angle.
        """
        # rotate the 3D array according to the rotation_angle
        if self._rot_angle == 90:
            img = np.flip(np.transpose(image, (1, 0, 2)), axis=0)
        elif self._rot_angle == 180:
            return np.flip(image, (0, 1))
        elif self._rot_angle == 270:
            img = np.flip(np.transpose(image, (1, 0, 2)), axis=1)
        else:
            img = image
        while img.ndim < 3:
            img = np.expand_dims(img, img.ndim)
        return img

    @abstractmethod
    def start_streaming(self):
        """start the data streaming."""
        if not self.connected:
            self.connect()
        self._streaming = True

    @abstractmethod
    def stop_streaming(self):
        """interrupt the data stream."""
        self._streaming = False

    @abstractmethod
    def connect(self):
        """setup the connection to the device."""
        pass

    @abstractmethod
    def disconnect(self):
        """interrupt the connection to the device."""
        pass

    def __init__(self, id: str):
        """
        constructor

        Parameters
        ----------
        id: str
            the name of the device
        """
        self._last_changed = Signal()
        self._id = id
        self.set_rotation_angle(0)
        self._fps = None
        self._last = None
        self._streaming = False
        self._recording = False
        self._connected = False


class OpticalDevice(Device):
    """Initialize an optical device object"""

    _device: cv2.VideoCapture  # type: ignore
    _reader: Thread
    _port: int

    @property
    def port(self):
        """return the connection port"""
        return self._port

    def __init__(self, device_id: str):
        """constructor"""
        super().__init__(device_id)
        self._device = None  # type: ignore
        self._reader = Thread(target=self._add_frame)

        # find the device
        devs = [
            i
            for i in self.get_available_devices()
            if not (i.lower().startswith("purethermal") or i == "PI IMAGER")
        ]
        valid_port = [i for i, v in enumerate(devs) if v == self.id]
        if len(valid_port) == 0:
            raise ValueError(f"No devices have been found with the {id} id.")
        self._port = valid_port[0]

    def _add_frame(self):
        """add a new frame to the buffer of readed data."""
        while self.streaming:
            img = self._device.read()[1]
            if img is not None:
                img = np.flip(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 1)
                self.set_last_sample(img)

    def start_streaming(self):
        """start the data streaming"""
        super().start_streaming()
        self._reader.start()
        while self._last is None:
            pass

    def stop_streaming(self):
        """stop the data streaming"""
        super().stop_streaming()
        if self._device.isOpened():
            self._device.release()

    def connect(self):
        """start the connection to the camera"""
        self._device = cv2.VideoCapture(self.port)
        self._connected = True

    def disconnect(self):
        """stop the connection with the camera"""
        if self.streaming:
            self.stop_streaming()
        self._device = None  # type: ignore
        self._connected = False


class LeptonDevice(Device):
    """
    Initialize a Lepton camera object capable of communicating to
    an pure thermal device equipped with a lepton 3.5 sensor.
    """

    _sensor: Any
    _stream: Any
    _reader: Any

    def __init__(self, id: str):
        """constructor"""
        super().__init__(id)

        # check whether python is running as 64bit or 32bit to import the right
        # .NET dll
        architecture = "x64" if platform.architecture()[0] == "64bit" else "x86"
        leptonuvc = join(LEPTON_PATH, architecture, "LeptonUVC")
        irmanager = join(LEPTON_PATH, architecture, "ManagedIR16Filters")
        clr.AddReference(leptonuvc)  # type: ignore
        clr.AddReference(irmanager)  # type: ignore

        from IR16Filters import IR16Capture, NewBytesFrameEvent
        from Lepton import CCI

        # find the device
        valid_device = [i for i in CCI.GetDevices() if i.Name == id]
        if len(valid_device) == 0:
            raise ValueError(f"No devices have been found with the {id} id.")
        self._sensor = valid_device[0]
        self._stream = None

        # setup the lepton data buffer
        # (this is used internally and should not replace the Device.buffer)
        self._reader = IR16Capture()
        callback = NewBytesFrameEvent(self._add_frame)
        self._reader.SetupGraphWithBytesCallback(callback)

    def _add_frame(self, array: bytearray, width: int, height: int):
        """
        add a new frame to the buffer of readed data.

        Parameters:
        array: bytearray
            the array of values

        width: int
            the width of the image

        height: int
            the height of the image
        """
        # get the thermal image
        img = np.fromiter(array, dtype="uint16").reshape((height, width))

        # centikelvin --> celsius conversion
        img = (img - 27315.0) / 100.0

        # make 3D
        img = np.expand_dims(img, 2)

        # mirror
        img = np.flip(img, 1)

        # update the last sampled data
        self.set_last_sample(img)

    def start_streaming(self):
        """start the data streaming"""
        super().start_streaming()
        self._reader.RunGraph()
        while self._last is None:
            pass

    def stop_streaming(self):
        """stop the data streaming"""
        self._reader.StopGraph()
        super().stop_streaming()

    def connect(self):
        """setup the device connection"""
        if not self.connected:
            self._stream = self._sensor.Open()
            self._stream.sys.RunFFCNormalization()
            self._stream.rad.SetTLinearEnableStateChecked(True)
        self._connected = True

    def disconnect(self):
        """interrupt the connection to the device"""
        if self.streaming:
            self.stop_streaming()
        if self._stream is not None:
            self._stream = self._stream.port.Dispose()
        self._connected = False


class OptrisPiDevice(Device):
    """Initialize an Optris PI camera."""

    _stream: Thread
    _device: ct.CDLL

    def __init__(self, id: str):
        """constructor"""
        super().__init__(id)
        if platform.architecture()[0] == "64bit":
            self._device = ct.CDLL(join(PI_PATH, "x64", "libirimager.dll"))
        else:
            self._device = ct.CDLL(join(PI_PATH, "x86", "libirimager.dll"))
        self._stream = Thread(target=self._add_frame)

    def _add_frame(self):
        """internal method used to read data from camera"""
        # init vars
        width = ct.c_int()
        height = ct.c_int()
        metadata = ct.byref(_EvoIRFrameMetadata())

        # get thermal image size
        self._device.evo_irimager_get_thermal_image_size(
            ct.byref(width),
            ct.byref(height),
        )

        # init thermal data container
        np_thermal = np.zeros(
            shape=(width.value, height.value),
            dtype=np.uint16,
            order="F",
        )
        np_pointer = np_thermal.ctypes.data_as(ct.POINTER(ct.c_ushort))

        # run
        while self._streaming:
            ret = self._device.evo_irimager_get_thermal_image_metadata(
                ct.byref(width),
                ct.byref(height),
                np_pointer,
                metadata,
            )
            if ret == 0:
                img = np.fliplr((np_thermal / 10 - 100).astype(float).T)
                img = np.expand_dims(img, 2)
                self.set_last_sample(img)

    def start_streaming(self):
        """start data streaming"""
        super().start_streaming()
        self._streaming = True
        self._stream.start()
        while self._last is None:
            pass

    def stop_streaming(self):
        """stop the data streaming"""
        super().stop_streaming()

    def connect(self):
        """setup the device connection"""
        if not self.connected:
            specs = join(PI_PATH, "generic.xml").encode()
            path = PI_PATH.encode()
            cwd = join("_optris", "pi", "logs")
            makedirs(cwd, exist_ok=True)
            cwd = join(cwd, "log").encode()
            ret = self._device.evo_irimager_usb_init(specs, path, cwd)
            self._connected = ret == 0
            if not self._connected:
                raise RuntimeError(f"Impossible to connect to {id}")
            else:
                remove(join(getcwd(), "log.txt"))

    def disconnect(self):
        """interrupt the connection to the device"""
        if self.streaming:
            self.stop_streaming()
        if self.connected:
            self._device.evo_irimager_terminate()
        self._connected = False


class _EvoIRFrameMetadata(ct.Structure):
    """
    private class used by OptrisPiDevice to store the data
    returned by the camera
    """

    _fields_ = [
        ("counter", ct.c_uint),
        ("counterHW", ct.c_uint),
        ("timestamp", ct.c_longlong),
        ("timestampMedia", ct.c_longlong),
        ("flagState", ct.c_int),
        ("tempChip", ct.c_float),
        ("tempFlag", ct.c_float),
        ("tempBox", ct.c_float),
    ]
