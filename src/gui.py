"""gui module"""

#! IMPORTS


import json
from collections import deque
from datetime import datetime
from os.path import dirname, exists, join, sep
from typing import Any, Dict, List, Tuple, Union, Literal

import cv2
import numpy as np
import qimage2ndarray
from numpy.typing import NDArray
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt

from .devices import *
from .assets import *


#! CONSTANTS


COLORMAPS = [
    " ".join(i.split("_")[1:]).capitalize()
    for i in cv2.__dict__
    if i.startswith("COLORMAP")
]
ICON_SIZE = 60


#! FUNCTIONS


def to_pixmap(b64: str, size: int = ICON_SIZE):
    """convert the input string into a pixmap"""
    barr = QtCore.QByteArray.fromBase64(bytes(b64, "utf-8"))
    pix = QtGui.QPixmap()
    pix.loadFromData(barr, "png")
    pix = pix.scaled(size, size)
    return pix


#! CLASSES


class HoverWidget(QtWidgets.QWidget):
    """
    defines a hover pane to be displayed over a matplotlib figure.
    """

    _labels: Dict[str, QtWidgets.QLabel]

    @property
    def labels(self):
        """return the stored labels"""
        return self._labels

    def __init__(self):
        super().__init__()
        layout = QtWidgets.QGridLayout()
        layout.setSpacing(5)
        self.setLayout(layout)
        flags = Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint  # type: ignore
        self.setWindowFlags(flags)
        self._labels = {}
        self.setStyleSheet("background-color: rgba(0, 0, 0, 100)")

    def add_label(
        self,
        name: str,
        unit: str,
    ):
        """
        add a new label to the hover.

        Parameters
        ----------
        name: str
            the name of the axis

        unit: str
            the unit of measurement to be displayed.
        """
        # check the entries
        assert isinstance(name, str), "name must be a str."
        assert isinstance(unit, str), "unit must be a str."

        # add the new label
        layout = self.layout()
        n = len(self.labels) + 1
        name_label = QtWidgets.QLabel(name)
        name_label.setAlignment(Qt.AlignVCenter | Qt.AlignRight)  # type: ignore
        name_label.setStyleSheet("color : red")
        layout.addWidget(name_label, n, 1)  # type: ignore
        self.labels[name] = QtWidgets.QLabel("")
        self.labels[name].setStyleSheet("color : red")
        self.labels[name].setAlignment(Qt.AlignVCenter | Qt.AlignCenter)  # type: ignore
        layout.addWidget(self.labels[name], n, 2)  # type: ignore
        unit_label = QtWidgets.QLabel(unit)
        unit_label.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)  # type: ignore
        unit_label.setStyleSheet("color : red")
        layout.addWidget(unit_label, n, 3)  # type: ignore
        self.setLayout(layout)

    def format_value(self, value: Union[int, float]):
        """
        format each value

        Parameters
        ----------
        value: int, float
            the value to be formatted

        Returns
        -------
        frm: str
            the formatted value
        """
        if value is None:
            return ""
        if isinstance(value, int):
            return f"{value:0.0f}"
        return f"{value:0.2f}"

    def update(self, **labels):
        """
        update the hover parameters.

        Parameters
        ----------
        labels: any
            keyworded values to be updated
        """
        for label, value in labels.items():
            if np.any([label == i for i in self.labels]):  # type: ignore
                vals = np.array([value]).flatten().tolist()
                txt = "\n".join(self.format_value(i) for i in vals)
                self.labels[label].setText(txt)


class ImageWidget(QtWidgets.QLabel):
    """render an image on a widget"""

    _data: Union[NDArray[Any], None]
    _pix: QtGui.QPixmap
    _fps: Union[float, None]
    _hover: HoverWidget
    _mouse_coords: Union[None, Tuple[int, int, int, int]]
    _colormap: int
    _ranges: Union[NDArray[Any], None]

    @property
    def hover(self):
        """return the hover widget"""
        return self._hover

    @property
    def data(self):
        """return the available data"""
        return self._data

    @property
    def fps(self):
        """return the actual fps"""
        return self._fps

    @property
    def ranges(self):
        """return the actual colormap ranges"""
        return self._ranges

    @property
    def mouse_coords(self):
        """return the available data"""
        return self._mouse_coords

    @property
    def pixmap(self):
        """return the rendered pixmap"""
        return self._pix

    def enterEvent(self, event: Union[None, QtGui.QEnterEvent] = None):
        """override enterEvent."""
        self.mouseMoveEvent(event)  # type: ignore

    def mouseMoveEvent(self, event: Union[None, QtGui.QMouseEvent] = None):
        """override moveEvent."""
        self._update_mouse_coords(event)

    def leaveEvent(self, event: Union[None, QtGui.QMouseEvent] = None):
        """override leaveEvent."""
        self.mouseMoveEvent(None)

    def _update_mouse_coords(
        self,
        event: Union[QtGui.QEnterEvent, QtGui.QMouseEvent, None],
    ):
        """
        extract mouse coordinates with respect to the available data, widget
        and screen

        Parameters
        ----------
        event : Union[QtGui.QEnterEvent, QtGui.QMouseEvent, None]
            the event generating the calculation of the mouse coordinates

        Returns
        -------
        coords: dict[str, tuple[int, int]] | None
            the coordinates of the mouse.
        """
        # if data is None, there is nothing to be reported
        if self._data is None or event is None or self._pix is None:
            self._mouse_coords = None  # type: ignore
        else:
            # get the position of the mouse in widget coordinates
            pos = event.localPos()  # type: ignore
            x_wdg = int(pos.x())
            y_wdg = int(pos.y())

            # get the pixmap size
            w_pix = self._pix.width()
            h_pix = self._pix.height()

            # get the widget size
            w_wdg = self.width()
            h_wdg = self.height()

            # check if the mouse is effectively within the image considering
            # that the pixmap lies in the middle of the image_widget space
            x_off = (w_wdg - w_pix) // 2
            y_off = (h_wdg - h_pix) // 2

            # get the true mouse coordinates of pix
            x_pix = x_wdg - x_off
            y_pix = y_wdg - y_off

            # if any of the offsets is negative or any of the the mouse coords
            # is greater than pix + off means that the mouse is within the
            # image_widget but outside the pixmap. Therefore no mouse coords
            # have to be provided
            if x_pix < 0 or y_pix < 0 or x_pix > w_pix or y_pix > h_pix:
                self._mouse_coords = None
            else:
                # get the scaling factor to be applied to pix coordinates to
                # obtain the corresponding data coordinates
                h_data, w_data = self._data.shape[:2]
                h_scale = h_data / h_pix
                w_scale = w_data / w_pix

                # obtain the mouse coords in data units
                x_data = max(0, min(w_data - 1, int(round(w_scale * x_pix))))
                y_data = max(0, min(h_data - 1, int(round(h_scale * y_pix))))

                # obtain the same coords in global units
                x_glb = event.globalX()
                y_glb = event.globalY()

                self._mouse_coords = (x_data, y_data, x_glb, y_glb)

        # update the hover
        self._update_hover()

    def _update_hover(self):
        """update the hover position"""
        if self._mouse_coords is None or self._data is None:
            self._hover.setVisible(False)
        else:
            self._hover.setVisible(True)

            # set the point where the hover shall be rendered
            off = 15
            x_data, y_data, x_glb, y_glb = self._mouse_coords
            self._hover.move(x_glb + off, y_glb + off)

            # update the values within the hover
            z_data = self._data[y_data, x_data]
            self._hover.update(x=x_data, y=y_data, z=z_data, fps=self._fps)

    def _update_view(self):
        """
        update the rendered image.

        Parameters
        ----------
        data: np.ndarray
            the new image

        fps: float
            the updating speed from the last sample
        """
        # udpate the image
        size = self.size()
        if self._data is None:
            img = np.random.randn(size.height(), size.width())
        else:
            img = self._data.copy()

        # apply the colormap if applicable
        if self.ranges is not None:
            minv, maxv = self.ranges
            img = ((img - minv) / (maxv - minv) * 255).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            img = cv2.applyColorMap(img, self._colormap)

        # convert to pixmap and store
        pix = QtGui.QPixmap.fromImage(qimage2ndarray.array2qimage(img))
        self._pix = pix.scaled(size, QtCore.Qt.AspectRatioMode.KeepAspectRatio)
        self.setPixmap(self._pix)

        # update the hover
        self._update_hover()

    def update(
        self,
        data: NDArray[Any],
        fps: float,
        colormap: int,
        ranges: Union[None, NDArray[Any]],
    ):
        """update the data to be viewed"""
        self._data = data
        self._fps = fps
        self._colormap = colormap
        self._ranges = ranges
        self._update_view()

    def __init__(self):
        super().__init__()

        # policies and alignment
        policy_exp = QtWidgets.QSizePolicy.Policy.Expanding

        # setup
        self.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.setSizePolicy(policy_exp, policy_exp)
        self.setMouseTracking(True)
        self._data = None
        self._fps = np.nan
        self._mouse_coords = None
        self._pix = None

        # set the hover
        self._hover = HoverWidget()
        self._hover.add_label("fps", "Hz")
        self._hover.add_label("x", "px")
        self._hover.add_label("y", "px")
        self._hover.add_label("z", "")
        self._hover.setVisible(False)


class LabelledSlider(QtWidgets.QWidget):
    """
    a slider linked to a label showing its value

    Parameters
    ----------
    value: int (optional, default=0)
        the default slider value

    minimum: int (optional, default=0)
        the default slider minimum value

    maximum: int (optional, default=1)
        the default slider maximum value

    step: int (optional, default=1)
        the default slider step value

    side: Literal['right', 'left']
        the side where to display the label
    """

    _label: QtWidgets.QTextEdit
    _slider: QtWidgets.QSlider
    _value_changed: Signal

    def __init__(
        self,
        value: int = 0,
        minimum: int = 0,
        maximum: int = 1,
        step: int = 1,
        side: Literal["right", "left"] = "left",
    ):
        super().__init__()
        layout = QtWidgets.QHBoxLayout()
        self._slider = QtWidgets.QSlider()
        self._label = QtWidgets.QTextEdit()
        self._label.textChanged.connect(self._on_label_change)
        self._value_changed = Signal()
        self._slider.valueChanged.connect(self._on_slider_change)
        self._slider.setOrientation(Qt.Orientation.Horizontal)
        self.set_minimum(minimum)
        self.set_maximum(maximum)
        self.set_value(value)
        self.set_step(step)
        if side == "right":
            layout.addWidget(self._slider)
            layout.addWidget(self._label)
        else:
            layout.addWidget(self._label)
            layout.addWidget(self._slider)
        self.setLayout(layout)
        self._label.setFixedWidth(ICON_SIZE)
        self.setFixedHeight(ICON_SIZE)
        self._slider.setMinimumWidth(2 * ICON_SIZE)

        # adjust the starting values
        self._on_slider_change()

    def _on_slider_change(self):
        """private method used to update label and trigger valueChanged events"""
        self._label.setText(str(self.value))
        cur = self._label.textCursor()
        cur.movePosition(
            QtGui.QTextCursor.Right,
            QtGui.QTextCursor.MoveAnchor,
            len(self._label.toPlainText()),
        )
        self._label.setTextCursor(cur)
        self._value_changed.emit(self.value)

    def _on_label_change(self):
        """private method used to update slider value"""
        try:
            new = int(self._label.toPlainText())
            self._slider.setValue(new)
        except Exception:
            pass

    def set_minimum(self, value: int):
        """set the minimum value of the slider"""
        self._slider.setMinimum(value)

    def set_maximum(self, value: int):
        """set the maximum value of the slider"""
        self._slider.setMaximum(value)

    def set_step(self, value: int):
        """set the step value of the slider"""
        self._slider.setSingleStep(value)

    def set_value(self, value: int):
        """set the minimum value of the slider"""
        self._slider.setValue(value)

    @property
    def minimum(self):
        """the minimum slider value"""
        return int(self._slider.minimum())

    @property
    def maximum(self):
        """the maximum slider value"""
        return int(self._slider.maximum())

    @property
    def step(self):
        """the single step slider value"""
        return int(self._slider.singleStep())

    @property
    def value(self):
        """the slider value"""
        return int(self._slider.value())

    @property
    def value_changed(self):
        """the value changed signal"""
        return self._value_changed


class RangeSlider(QtWidgets.QWidget):
    """
    generate a range slider

    Parameters
    ----------
    value: Tuple[int, int] (optional, default=(0, 3))
        the default slider range

    minimum: int (optional, default=0)
        the default slider minimum value

    maximum: int (optional, default=3)
        the default slider maximum value

    step: int (optional, default=1)
        the default slider step value

    active: bool (optional, default=True)
        should the range be active?
    """

    _left: LabelledSlider
    _right: LabelledSlider
    _active: QtWidgets.QCheckBox
    _value_changed: Signal

    def __init__(
        self,
        value: Tuple[int, int] = (0, 3),
        minimum: int = 0,
        maximum: int = 2,
        step: int = 1,
        active: bool = True,
    ):
        super().__init__()
        self._value_changed = Signal()
        layout = QtWidgets.QHBoxLayout()
        self._left = LabelledSlider(
            value=value[0],
            minimum=minimum,
            maximum=maximum - 2,
            step=step,
            side="left",
        )
        self._right = LabelledSlider(
            value=value[1],
            minimum=maximum - 1,
            maximum=maximum,
            step=step,
            side="right",
        )
        self._active = QtWidgets.QCheckBox("Active")
        self._active.setChecked(active)
        self._active.stateChanged.connect(self._check_pressed)
        self._active.setFixedHeight(ICON_SIZE)
        layout.addWidget(self._left)
        layout.addWidget(self._right)
        layout.addWidget(self._active)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 5, 0)
        self.setLayout(layout)
        self.setMinimumSize(self.minimumSizeHint())

        # adjust the values
        self._left.value_changed.connect(self._on_value_change)
        self._right.value_changed.connect(self._on_value_change)
        self._on_value_change()
        self._check_pressed()

    def _on_value_change(self, *args, **kwargs):
        """internal method used to handle the value update"""
        vall = self._left.value
        valr = self._right.value
        self._left.set_maximum(valr - 1)
        self._right.set_minimum(vall + 1)
        return self._value_changed.emit(self.value)

    def _check_pressed(self):
        """handle the press of the active checkbox"""
        self._left.setEnabled(self.is_active())
        self._right.setEnabled(self.is_active())

    def is_active(self):
        """return True if the actual object is active"""
        return bool(self._active.isChecked())

    def set_minimum(self, value: int):
        """set the minimum value of the slider"""
        self._left.set_minimum(value)

    def set_maximum(self, value: int):
        """set the maximum value of the slider"""
        self._right.set_maximum(value)

    def set_step(self, value: int):
        """set the step value of the slider"""
        self._left.set_step(value)
        self._right.set_step(value)

    def set_value(self, value: Tuple[int, int]):
        """set the minimum value of the slider"""
        self._left.set_value(value[0])
        self._right.set_value(value[1])

    def set_label(self, value: str):
        """set the RangeBox label"""
        self._active.setText(value)
        self.setFixedWidth(self.minimumSizeHint().width())

    @property
    def minimum(self):
        """the minimum slider value"""
        return int(self._left.minimum())

    @property
    def maximum(self):
        """the maximum slider value"""
        return int(self._right.maximum())

    @property
    def step(self):
        """the single step slider value"""
        return int(self._left.singleStep())

    @property
    def value(self):
        """the slider value"""
        return (self._left.value, self._right.value)

    @property
    def value_changed(self):
        """the value changed signal"""
        return self._value_changed


class RangeBox(QtWidgets.QWidget):
    """
    custom widget allowing to define the minimum and maximum values
    accepted for each color/value channel of an image

    Parameters
    ----------
    channels_n: int
        the number of channels

    minimum: int | float, (optional, default = -255)
        the minimum acceptable value

    maximum: int | float, (optional, default = 255)
        the maximum acceptable value

    orientation: Literal['vertical', 'horizontal'], (optional, default='vertical')
        the orientation of each box
    """

    _channels: List[RangeSlider]

    def __init__(
        self,
        channels_n: int,
        minimum: int = -255,
        maximum: int = 255,
        orientation: Literal["vertical", "horizontal"] = "horizontal",
    ):
        super().__init__()
        # setup the channels
        self._channels = []
        if orientation == "vertical":
            layout = QtWidgets.QVBoxLayout()
        else:
            layout = QtWidgets.QHBoxLayout()
        for i in range(channels_n):
            channel = RangeSlider()
            channel.set_minimum(minimum)
            channel.set_maximum(maximum)
            channel.set_value((minimum, maximum))
            channel.set_step(1)
            channel.set_label("chn " + str(i + 1))
            self._channels += [channel]
            layout.addWidget(channel)

        # set the widget layout
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)
        self.setFixedSize(self.minimumSizeHint())

    @property
    def value(self):
        """return the actual range values of each channel"""
        out = []
        for channel in self._channels:
            out += [channel.value if channel.is_active() else [None, None]]
        return np.atleast_2d(out).astype(float)

    @property
    def active_state(self):
        """return the actual active state of each channel"""
        return [channel.is_active() for channel in self._channels]


class CameraWidget(QtWidgets.QWidget):
    """
    Initialize a widget communicating to a camera device being capable of
    showing the captured data.

    Parameters
    ----------
    device: Device
        the data source
    """

    _timer: QtCore.QTimer
    _image_widget: ImageWidget
    _rotation_widget: QtWidgets.QPushButton
    _mirror_widget: QtWidgets.QPushButton
    _close_widget: QtWidgets.QPushButton
    _device: Device
    _closed: Signal
    _data: Union[None, NDArray[Any]]
    _fps: Union[None, float]
    _colormap: int
    _rangebox: RangeBox

    @property
    def rangebox(self):
        """return the rangebox"""
        return self._rangebox

    @property
    def device(self):
        """return the device linked to this object"""
        return self._device

    @property
    def closed(self):
        """closing signal"""
        return self._closed

    def start_streaming(self):
        """start running the camera"""
        self.device.start_streaming()
        self._timer.stop()
        self._timer.start(int(round(1000 / 30)))

    def show(self):
        """overrides the widget show function"""
        self.start_streaming()
        super().show()

    def stop(self):
        """stop the streadming"""
        self.device.disconnect()
        self._timer.stop()

    def setEnabled(self, enabled: bool):
        """extend enabling/disabling to all childs"""
        self._rotation_widget.setEnabled(enabled)
        self._close_widget.setEnabled(enabled)

    def _rotate_image(self):
        """function used to rotate the image"""
        self.device.set_rotation_angle((self.device.rotation + 90) % 360)

    def _mirror_image(self):
        """function used to mirror the image"""
        self.device.set_mirroring(self._mirror_widget.isChecked())

    def _close(self):
        """close the widget"""
        self.device.disconnect()
        self._closed.emit(self.device.id)
        self.close()

    def _update_view(self):
        """update the data to be viewed"""
        if self._data is not None:
            img = self._data.copy()

            # filter the image according to the actual imposed ranges
            rngs = []
            for i, vals in enumerate(self.rangebox.value):
                minv, maxv = vals
                if not np.isnan(minv):
                    xi, yi, zi = np.where(img < minv)
                    ki = np.where(zi == i)[0]
                    img[xi[ki], yi[ki], zi[ki]] = minv
                if not np.isnan(maxv):
                    xi, yi, zi = np.where(img > maxv)
                    ki = np.where(zi == i)[0]
                    img[xi[ki], yi[ki], zi[ki]] = maxv
                rngs += [np.min(img), np.max(img)]
            rngs = np.squeeze(rngs) if img.shape[-1] == 1 else None

            # update the image
            self._image_widget.update(img, self._fps, self._colormap, rngs)

    def _update_data(self):
        """update the available data"""
        if self.device.last_sample is not None:
            self._data = self.device.last_sample[1]
            self._fps = self.device.fps
        else:
            self._data = None
            self._fps = None

    def update_colormap(self, colormap: int):
        """update the actual colormap"""
        self._colormap = colormap

    def __init__(self, device: Device):
        super().__init__()

        # setup the camera device
        self._device = device
        self._device.last_changed.connect(self._update_data)

        # setup data and fps retrieved from device
        self._data = None
        self._fps = None

        # setup the data refreshing timer
        self._timer = QtCore.QTimer()
        self._timer.timeout.connect(self._update_view)

        # setup signals
        self._closed = Signal()

        # camera rotation
        rotation_icon = QtGui.QIcon(to_pixmap(ROTATE))
        self._rotation_widget = QtWidgets.QPushButton()
        self._rotation_widget.setIcon(rotation_icon)
        self._rotation_widget.clicked.connect(self._rotate_image)
        rot_wdg = OptionWidget(
            widgets=[self._rotation_widget],
            label="",
            tooltip="Rotate the image clockwise by 90 degrees.",
        )

        # camera mirroring
        mirror_icon = QtGui.QIcon(to_pixmap(MIRROR))
        self._mirror_widget = QtWidgets.QPushButton()
        self._mirror_widget.setIcon(mirror_icon)
        self._mirror_widget.setCheckable(True)
        self._mirror_widget.clicked.connect(self._mirror_image)
        mir_wdg = OptionWidget(
            widgets=[self._mirror_widget],
            label="",
            tooltip="Mirror the image.",
        )

        # close widget
        close_icon = to_pixmap(CLOSE)
        self._close_widget = QtWidgets.QPushButton()
        self._close_widget.setIcon(QtGui.QIcon(close_icon))
        self._close_widget.clicked.connect(self._close)
        cls_wdg = OptionWidget(
            widgets=[self._close_widget],
            label="",
            tooltip="Close the communication with this camera.",
        )

        # label widget
        label = QtWidgets.QLabel(self._device.id)

        # rangebox
        n_channels = 3 if isinstance(self.device, OpticalDevice) else 1
        maxv = 255 if n_channels == 3 else 273
        minv = 0 if n_channels == 3 else -273
        self._rangebox = RangeBox(
            channels_n=n_channels,
            minimum=minv,
            maximum=maxv,
            orientation="vertical",
        )

        # setup the options panel
        linew = QtWidgets.QWidget()
        linel = QtWidgets.QHBoxLayout()
        linel.addWidget(label)
        linel.addWidget(rot_wdg)
        linel.addWidget(mir_wdg)
        linel.addWidget(cls_wdg)
        linel.setSpacing(0)
        linel.setContentsMargins(0, 0, 0, 0)
        linel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        linew.setLayout(linel)
        linew.setFixedHeight(linew.minimumSizeHint().height())
        opt_layout = QtWidgets.QVBoxLayout()
        opt_layout.addWidget(linew)
        opt_layout.addWidget(self._rangebox)
        opt_layout.addStretch()
        opt_layout.setSpacing(0)
        opt_layout.setContentsMargins(0, 0, 0, 0)
        opt_wdg = QtWidgets.QWidget()
        opt_wdg.setLayout(opt_layout)
        opt_wdg.setFont(QtGui.QFont("Arial", 10))
        opt_width = max(
            self._rangebox.minimumSizeHint().width(),
            linew.minimumSizeHint().width(),
        )
        opt_wdg.setFixedWidth(opt_width)

        # image panel
        self._image_widget = ImageWidget()
        self._image_widget.setMinimumHeight(opt_wdg.sizeHint().height())

        # widget layout
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self._image_widget)
        layout.addWidget(opt_wdg)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)


class DeviceDialog(QtWidgets.QDialog):
    """dialog used to add novel devices"""

    _box: QtWidgets.QComboBox
    _devices: List[str]
    _button_ok: QtWidgets.QPushButton
    _button_cancel: QtWidgets.QPushButton
    _button_all: QtWidgets.QPushButton
    _add_clicked: Signal
    _all_clicked: Signal

    @property
    def all_clicked(self):
        """return the all_clicked signal"""
        return self._all_clicked

    @property
    def add_clicked(self):
        """return the add_clicked signal"""
        return self._add_clicked

    def __init__(self, device_ids: List[str]):
        super().__init__()
        self._devices = device_ids
        self.setModal(True)
        self.setWindowTitle("Add a new device")

        # setup the combobox
        label = QtWidgets.QLabel("Available devices: ")
        self._box = QtWidgets.QComboBox(self)
        for device in device_ids:
            self._box.addItem(device)
        self._box.setFixedWidth(self._box.minimumSizeHint().width())
        box_layout = QtWidgets.QHBoxLayout()
        box_layout.addWidget(label)
        box_layout.addWidget(self._box)
        box_pane = QtWidgets.QWidget()
        box_pane.setLayout(box_layout)

        # setup the buttons
        self._button_all = QtWidgets.QPushButton("Add All")
        self._button_ok = QtWidgets.QPushButton("Add")
        self._button_cancel = QtWidgets.QPushButton("Cancel")
        min_button_ok_width = self._button_ok.minimumSizeHint().width()
        min_button_cancel_width = self._button_cancel.minimumSizeHint().width()
        button_width = min(min_button_ok_width, min_button_cancel_width)
        self._button_ok.setFixedWidth(button_width)
        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(self._button_all)
        button_layout.addStretch()
        button_layout.addWidget(self._button_ok)
        button_layout.addWidget(self._button_cancel)
        button_pane = QtWidgets.QWidget()
        button_pane.setLayout(button_layout)

        # setup the UI
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(box_pane)
        layout.addWidget(button_pane)
        self.setLayout(layout)

        # add the callbacks
        self._button_ok.clicked.connect(self._add_selected)
        self._add_clicked = Signal()
        self._button_cancel.clicked.connect(self._cancel_selected)
        self._button_all.clicked.connect(self._all_selected)
        self._all_clicked = Signal()

    def _add_selected(self):
        """handle the click of the add button"""
        self._add_clicked.emit(self._box.currentText())
        self.close()

    def _all_selected(self):
        """handle the click of the add all button"""
        self._all_clicked.emit(self._devices)
        self.close()

    def _cancel_selected(self):
        """handle the click of the cancel button"""
        self.close()


class OptionWidget(QtWidgets.QWidget):
    """
    option widget

    Parameters
    ----------
    widgets : List[QtWidgets.QWidget]
        list of necessary widgets to be included

    label : str
        description of the widget type

    tooltip : str
        tooltip of the final widget
    """

    _widgets: List[QtWidgets.QWidget]
    _label: QtWidgets.QLabel

    def __init__(
        self,
        widgets: List[QtWidgets.QWidget],
        label: str,
        tooltip: str,
    ):
        """make option widgets"""

        # generate the output layout
        lay = QtWidgets.QHBoxLayout()
        self._label = QtWidgets.QLabel(label)
        self._widgets = []
        for i, wdg in enumerate([self._label] + widgets):
            wdg.setFont(QtGui.QFont("Arial", 12))
            lay.addWidget(wdg)
            if i > 0:
                self._widgets += [wdg]

        # create the output widget
        super().__init__()
        lay.setSpacing(0)
        lay.setContentsMargins(5, 0, 5, 0)
        self.setLayout(lay)
        self.setToolTip(tooltip)
        self.setFixedHeight(self.minimumSizeHint().height())

    @property
    def widgets(self):
        return self._widgets

    @property
    def label(self):
        return self._label


class IRCam(QtWidgets.QMainWindow):
    """
    IRCam Application

    Generate a Graphical interface capable of connecting and recording
    data simultaneously from multiple video devices.
    """

    # general variables
    _path: str
    _configuration: Dict[str, Any]
    _buffer: Dict[CameraWidget, deque]
    _recording_handler: QtCore.QTimer

    # add camera objects
    _add_button: QtWidgets.QPushButton
    _add_wdg: OptionWidget

    # recording objects
    _start_time: datetime
    _last_time: datetime
    _proc_time: float
    _rec_freq_spinbox: QtWidgets.QSpinBox
    _frq_wdg: OptionWidget
    _time_label_format = "{:02d}:{:02d}:{:02d}"
    _time_label: QtWidgets.QLabel
    _rec_button: QtWidgets.QPushButton
    _rec_icon: QtGui.QIcon
    _stop_icon: QtGui.QIcon
    _save_button: QtWidgets.QPushButton
    _rec_wdg: OptionWidget
    _save_wdg: OptionWidget

    # colormap
    _colormap_box: QtWidgets.QComboBox
    _colormap_wdg: OptionWidget

    # container
    _container: QtWidgets.QVBoxLayout

    def __init__(self):
        super().__init__()

        # retrieve the configuration file
        if not exists(self.configuration_file):
            self._configuration = {
                "sfrq": 6,
                "path": dirname(__file__),
                "cmap": COLORMAPS[0],
            }
        else:
            with open(self.configuration_file, "r") as buf:
                self._configuration = json.load(buf)

        # variables init
        self._buffer = {}
        self._path = self.config["path"]
        self._last_time = datetime.now()
        self._recording_handler = QtCore.QTimer()
        self._recording_handler.timeout.connect(self._recorder)

        # icons
        self._rec_icon = QtGui.QIcon(to_pixmap(REC))
        self._stop_icon = QtGui.QIcon(to_pixmap(STOP))
        save_icon = QtGui.QIcon(to_pixmap(SAVE))
        add_icon = QtGui.QIcon(to_pixmap(ADD))
        main_icon = QtGui.QIcon(to_pixmap(MAIN))

        # size policies
        policy_exp = QtWidgets.QSizePolicy.Policy.Expanding

        # container widget
        self._container = QtWidgets.QVBoxLayout()
        self._container.setSpacing(15)
        self._container.setContentsMargins(0, 0, 0, 0)
        self._container.setAlignment(Qt.AlignmentFlag.AlignTop)
        cnt_widget = QtWidgets.QWidget()
        cnt_widget.setLayout(self._container)
        cnt_widget.setSizePolicy(policy_exp, policy_exp)

        # recording frequency widget
        self._rec_freq_spinbox = QtWidgets.QSpinBox()
        self._rec_freq_spinbox.setValue(self.config["sfrq"])
        self._rec_freq_spinbox.setMaximum(30)
        self._rec_freq_spinbox.setMinimum(1)
        self._rec_freq_spinbox.setSingleStep(1)
        self._rec_freq_spinbox.setSuffix("Hz")
        self._frq_wdg = OptionWidget(
            widgets=[self._rec_freq_spinbox],
            label="Frequency",
            tooltip="Adjust the recording frequency from the cameras.",
        )

        # recording button with time label
        self._rec_button = QtWidgets.QPushButton()
        self._rec_button.setIcon(self._rec_icon)
        self._rec_button.setCheckable(True)
        self._rec_button.clicked.connect(self._rec_button_pressed)
        time_label = self._time_label_format.format(0, 0, 0)
        self._time_label = QtWidgets.QLabel(time_label)
        self._rec_wdg = OptionWidget(
            widgets=[self._rec_button, self._time_label],
            label="",
            tooltip="Start and stop the data recording.",
        )

        # save widget
        self._save_button = QtWidgets.QPushButton()
        self._save_button.setIcon(save_icon)
        self._save_button.clicked.connect(self._save_button_pressed)
        self._save_wdg = OptionWidget(
            widgets=[self._save_button],
            label="Save",
            tooltip="Save the recorded data",
        )

        # recording object
        rec_layout = QtWidgets.QHBoxLayout()
        rec_layout.addWidget(self._frq_wdg)
        rec_layout.addStretch()
        rec_layout.addWidget(self._rec_wdg)
        rec_layout.addStretch()
        rec_layout.addWidget(self._save_wdg)
        rec_obj = QtWidgets.QWidget()
        rec_layout.setSpacing(5)
        rec_layout.setContentsMargins(0, 0, 0, 0)
        rec_obj.setLayout(rec_layout)

        # colormap widget
        self._colormap_box = QtWidgets.QComboBox()
        self._colormap_box.addItems(COLORMAPS)
        cmap = [i for i, v in enumerate(COLORMAPS) if v == self.config["cmap"]]
        self._colormap_box.setCurrentIndex(int(cmap[0]))
        self._colormap_box.setFixedWidth(200)
        self._colormap_wdg = OptionWidget(
            widgets=[self._colormap_box],
            label="Colormap",
            tooltip="Change the colormap applied to non RGB cameras.",
        )
        self._colormap_box.currentIndexChanged.connect(self._update_colormaps)

        # add widget
        self._add_button = QtWidgets.QPushButton()
        self._add_button.setIcon(add_icon)
        self._add_button.clicked.connect(self._add_button_pressed)
        self._add_wdg = OptionWidget(
            widgets=[self._add_button],
            label="Add",
            tooltip="Add a new device.",
        )

        # setup the options layout
        opt_layout = QtWidgets.QHBoxLayout()
        opt_layout.addStretch()
        opt_layout.addWidget(rec_obj)
        opt_layout.addStretch()
        opt_layout.addWidget(self._colormap_wdg)
        opt_layout.addStretch()
        opt_layout.addWidget(self._add_wdg)
        opt_layout.addStretch()
        opt_layout.setSpacing(5)
        opt_layout.setContentsMargins(5, 15, 5, 5)
        opt_wdg = QtWidgets.QWidget()
        opt_wdg.setLayout(opt_layout)
        opt_wdg.setFixedHeight(opt_wdg.minimumSizeHint().height())

        # setup the widget layout
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(cnt_widget)
        layout.addWidget(opt_wdg)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        central_widget = QtWidgets.QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        # window setup
        self._check_enabled()
        self.setWindowTitle("IRCam")
        self.setWindowIcon(main_icon)
        self.setStyleSheet("background-color: white")
        self.setFont(QtGui.QFont("Arial", 12))
        self.setMinimumWidth(self.minimumSizeHint().width())

    @property
    def config(self):
        """return the configuration file"""
        return self._configuration

    @property
    def configuration_file(self):
        """return the configuration file"""
        return join("C:", "_bbconf.json")

    def _update_colormaps(self):
        """update the colormap of all the available cameras"""
        cmap = self._colormap_box.currentIndex()
        for cam in list(self._buffer.keys()):
            cam.update_colormap(cmap)

    def _check_enabled(self):
        """
        private method used to check whether widgets have to be enabled or
        disabled
        """
        # get the testing conditions
        rec_checked = self._rec_button.isChecked()
        cam_exist = len(self._buffer) > 0
        if len(self._buffer.values()) > 0:
            data_exist = any(len(v) > 0 for v in self._buffer.values())
        else:
            data_exist = False
        rgb_cam_only = all(
            [isinstance(i.device, OpticalDevice) for i in self._buffer.keys()]
        )

        # apply
        self._save_wdg.setEnabled(~rec_checked & data_exist)
        self._frq_wdg.setEnabled(~rec_checked & cam_exist)
        self._rec_wdg.setEnabled(cam_exist)
        self._colormap_wdg.setEnabled(cam_exist & ~rgb_cam_only)
        self._add_wdg.setEnabled(~rec_checked)
        for cam in list(self._buffer.keys()):
            cam.setEnabled(~rec_checked)

    def _recorder(self):
        """function used to handle the data recording from multiple devices"""
        # check if new data have to be collected
        target = 1 / int(self._rec_freq_spinbox.value())
        now = datetime.now()
        delta = (now - self._last_time).total_seconds()
        if delta + self._proc_time >= target:
            for i, cam in enumerate(list(self._buffer.keys())):
                last = cam.device.last_sample
                if last is not None:
                    self._buffer[cam].append((now, last[-1]))
                    if i == 0 and len(self._buffer[cam]) > 1:
                        proc = (delta - target + 2 * self._proc_time) / 2
                        self._proc_time = proc
            self._last_time = now

        # update the recording label
        delta = (now - self._start_time).total_seconds()
        h = int(delta // 3600)
        m = int((delta - h * 3600) // 60)
        s = int((delta - h * 3600 - m * 60) // 1)
        self._time_label.setText(self._time_label_format.format(h, m, s))

    def _rec_button_pressed(self):
        """handle the clicking of the recording button"""
        # handle specific checked / unchecked actions
        if self._rec_button.isChecked():
            for cam in list(self._buffer.keys()):
                self._buffer[cam] = deque()
            self._rec_button.setIcon(self._stop_icon)
            self._start_time = datetime.now()
            self._proc_time = 0
            self._recording_handler.start(10)
        else:
            self._recording_handler.stop()
            while self._recording_handler.isActive():
                pass
            self._time_label.setText(self._time_label_format.format(0, 0, 0))
            self._rec_button.setIcon(self._rec_icon)
            if any(len(v) > 0 for v in self._buffer.values()):
                self._save_button_pressed()
        self._check_enabled()

    def _save_button_pressed(self):
        """handle the click on the save button"""
        # start the data saving procedure
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select the folder where to save the collected data",
            self._path,
        )
        if path != "":
            path = path.replace("/", sep)
            self._path = path
            diag = QtWidgets.QProgressDialog(
                "",
                "Cancel",
                0,
                len(self._buffer),
                self,
            )
            diag.setWindowTitle("Saving data")
            diag.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
            diag.setFixedWidth(400)
            diag.forceShow()
            diag.setVisible(True)
            diag.setValue(0)
            for cam, buf in self._buffer.items():
                diag.setLabelText(cam.device.id)
                data = {i.strftime(TIMESTAMP_FORMAT): v for i, v in buf}
                time = list(data.keys())[0].rsplit(".", 1)[0].replace(" ", "_")
                time = time.replace("-", "").replace(":", "")
                name = cam.device.id.replace("(", "").replace(")", "")
                name = name.replace(".", "").replace(" ", "_")
                name = name.replace(":", "").replace("-", "").lower()
                file = join(path, "-".join([time, name]))
                np.savez_compressed(file + ".npz", **data)
                shape = list(data.values())[0].shape
                shape = shape[:-1][::-1] + (shape[-1],)
                writer = cv2.VideoWriter(
                    file + ".avi",
                    cv2.VideoWriter_fourcc(*"MJPG"),  # type: ignore
                    int(self._rec_freq_spinbox.value()),
                    shape[:2],
                    shape[-1] > 1,
                )
                for frame in data.values():
                    img = (frame / frame.max(axis=(0, 1)) * 255).astype(np.uint8)
                    if shape[-1] > 1:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    writer.write(img)
                writer.release()
                diag.setValue(diag.value() + 1)
                if diag.wasCanceled():
                    break
        self._check_enabled()

    def _add_button_pressed(self):
        """handle the pressure of the add button"""
        av_devices = self._usable_devices()
        if len(av_devices) > 0:
            diag = DeviceDialog(av_devices)
            diag.add_clicked.connect(self._add_single_camera)
            diag.all_clicked.connect(self._add_all_cameras)
            diag.exec_()
        else:
            QtWidgets.QMessageBox.information(
                self,
                "Add new device.",
                "Available devices not found.",
                QtWidgets.QMessageBox.StandardButton.Close,
            )
        self._check_enabled()

    def _usable_devices(self):
        """get the list of available devices"""
        devs = Device.get_available_devices()
        ids = [i.device.id for i in list(self._buffer.keys())]
        devs = [i for i in devs if i not in ids]
        return devs

    def _add_camera(self, id: str):
        """
        add a new camera to the container

        Parameters
        ----------
        id : str
            the name of the device to be included
        """

        # get the proper device
        if id.startswith("PureThermal"):
            new_device = LeptonDevice(id)
        elif id == "PI IMAGER":
            new_device = OptrisPiDevice(id)
        else:
            new_device = OpticalDevice(id)

        # generate the camera widget object and add it to the device list
        camera = CameraWidget(new_device)
        camera.closed.connect(self._remove_camera)
        camera.update_colormap(self._colormap_box.currentIndex())
        camera.start_streaming()
        self._buffer[camera] = deque()

    def _add_single_camera(self, id: str):
        """
        add a new camera to the container

        Parameters
        ----------
        id : str
            the name of the device to be included
        """
        self._add_camera(id)
        self._update_view()

    def _add_all_cameras(self, ids: List[str]):
        """
        add all available cameras

        Parameters
        ----------
        ids : List[str]
            list with the names of the device to be included
        """
        diag = QtWidgets.QProgressDialog("", "Cancel", 0, len(ids) + 1, self)
        diag.setWindowTitle("Adding devices")
        diag.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        diag.setFixedWidth(400)
        diag.forceShow()
        diag.setVisible(True)
        diag.setValue(0)
        for id in ids:
            diag.setLabelText(id)
            self._add_camera(id)
            diag.setValue(diag.value() + 1)
            if diag.wasCanceled():
                break
        self._update_view()
        diag.setValue(diag.value() + 1)

    def _update_view(self):
        """
        update the container layout according to the available number of
        cameras.
        """
        # clean the actual container layout
        for row in np.arange(self._container.count()):
            item = self._container.itemAt(row)
            self._container.removeItem(item)

        # refresh the layout
        cams = list(self._buffer.keys())
        if len(cams) > 0:
            for cam in cams:
                self._container.addWidget(cam)
        self._check_enabled()

    def _remove_camera(self, id: str):
        """
        remove the camera defined by the given id

        Parameters
        ----------
        id: str
            the id of the camera
        """
        idx = [i for i in list(self._buffer.keys()) if i.device.id == id]
        if len(idx) > 0:
            idx[0].close()
            self._buffer.pop(idx[0])
        self._update_view()

    def closeEvent(self, event: QtGui.QCloseEvent):
        """
        override the close event forcing the disconnection from all devices
        """
        # ensure all devices are disconnected
        for cam in list(self._buffer.keys()):
            cam.device.disconnect()

        # save the configuration
        self._configuration = {
            "sfrq": int(self._rec_freq_spinbox.value()),
            "path": self._path,
            "cmap": self._colormap_box.currentText(),
        }

        with open(self.configuration_file, "w") as buf:
            json.dump(self.config, buf)

        # close
        super().closeEvent(event)


__all__ = ["IRCam"]
