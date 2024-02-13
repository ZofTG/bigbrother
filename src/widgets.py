"""gui module"""

#! IMPORTS


from collections import deque
from datetime import datetime
import json
from os.path import dirname, join, sep, exists
from typing import Any, Dict, List, Tuple, Union

import cv2
import numpy as np
import qimage2ndarray
from numpy.typing import NDArray
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt

from . import devices

__all__ = ["BigBrother"]


#! CONSTANTS


COLORMAPS = [
    " ".join(i.split("_")[1:]).capitalize()
    for i in cv2.__dict__
    if i.startswith("COLORMAP")
]
ICON_PATH = join(dirname(dirname(__file__)), "icons")
ICON_SIZE = 50


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
    _pix: Union[None, QtGui.QPixmap]
    _fps: Union[float, None]
    _hover: HoverWidget
    _mouse_coords: Union[None, Tuple[int, int, int, int]]
    _colormap: int

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
            # check a colormap has to be applied to the data
            if self._data.shape[-1] == 1:
                # commute the available data into a greyscale image
                minv = np.min(self._data)
                img = self._data - minv
                img /= np.max(self._data) - minv
                img = (img * 255).astype(np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

                # apply the colormap
                img = cv2.applyColorMap(img, self._colormap)
            else:
                img = self._data.copy()

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
    ):
        """update the data to be viewed"""
        self._data = data
        self._fps = fps
        self._colormap = colormap
        self._update_view()

    def __init__(self):
        super().__init__()

        # policies and alignment
        policy_exp = QtWidgets.QSizePolicy.Policy.MinimumExpanding
        halignment = QtCore.Qt.AlignmentFlag.AlignHCenter
        valignment = QtCore.Qt.AlignmentFlag.AlignVCenter

        # setup
        self.setSizePolicy(policy_exp, policy_exp)
        self.setAlignment(halignment | valignment)  # type: ignore
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
    _close_widget: QtWidgets.QPushButton
    _device: devices.Device
    _closed: devices.Signal
    _data: Union[None, NDArray[Any]]
    _fps: Union[None, float]
    _colormap: int

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
        self.device.set_rotation_angle((self.device.rot_angle + 90) % 360)

    def _close(self):
        """close the widget"""
        self.device.disconnect()
        self._closed.emit(self.device.id)
        self.close()

    def _update_view(self):
        """update the data to be viewed"""
        self._image_widget.update(self._data, self._fps, self._colormap)

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

    def __init__(self, device: devices.Device):
        super().__init__()

        # setup the camera device
        if not isinstance(device, devices.Device):
            raise TypeError("the passed device must be a Device instance.")
        self._device = device
        self._device.last_changed.connect(self._update_data)

        # setup data and fps retrieved from device
        self._data = None
        self._fps = None

        # setup the data refreshing timer
        self._timer = QtCore.QTimer()
        self._timer.timeout.connect(self._update_view)

        # setup signals
        self._closed = devices.Signal()

        # camera rotation
        rotation_path = join(ICON_PATH, "rotate.png")
        rotation_icon = QtGui.QPixmap(rotation_path)
        rotation_icon = rotation_icon.scaled(ICON_SIZE, ICON_SIZE)
        self._rotation_widget = QtWidgets.QPushButton()
        self._rotation_widget.setIcon(QtGui.QIcon(rotation_icon))
        self._rotation_widget.clicked.connect(self._rotate_image)
        rot_wdg = OptionWidget(
            widgets=[self._rotation_widget],
            label="",
            tooltip="Rotate the image clockwise by 90 degrees.",
        )

        # close widget
        close_path = join(ICON_PATH, "close.png")
        close_icon = QtGui.QPixmap(close_path)
        close_icon = close_icon.scaled(ICON_SIZE, ICON_SIZE)
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

        # policies and alignment
        policy_exp = QtWidgets.QSizePolicy.Policy.MinimumExpanding
        policy_min = QtWidgets.QSizePolicy.Policy.Minimum
        central_alignment = QtCore.Qt.AlignmentFlag.AlignHCenter
        right_alignment = QtCore.Qt.AlignmentFlag.AlignRight
        vert_alignment = QtCore.Qt.AlignmentFlag.AlignVCenter
        bottom_alignment = QtCore.Qt.AlignmentFlag.AlignBottom

        # setup the options panel
        opt_layout = QtWidgets.QHBoxLayout()
        opt_layout.addStretch()
        opt_layout.addWidget(label)
        opt_layout.addWidget(rot_wdg)
        opt_layout.addWidget(cls_wdg)
        opt_layout.addStretch()
        opt_layout.setAlignment(central_alignment)  # type: ignore
        opt_layout.setSpacing(0)
        opt_layout.setContentsMargins(0, 0, 0, 0)
        opt_wdg = QtWidgets.QWidget()
        opt_wdg.setLayout(opt_layout)
        opt_wdg.setSizePolicy(policy_min, policy_min)
        opt_wdg.setFont(QtGui.QFont("Arial", 10))
        label.setSizePolicy(policy_exp, policy_min)
        label.setAlignment(right_alignment | vert_alignment)  # type: ignore

        # image panel
        self._image_widget = ImageWidget()
        self._image_widget.setAlignment(central_alignment | bottom_alignment)  # type: ignore

        # widget layout
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self._image_widget)
        layout.addWidget(opt_wdg)
        self.setLayout(layout)
        self.setSizePolicy(policy_exp, policy_exp)
        self.setMinimumHeight(100)


class DeviceDialog(QtWidgets.QDialog):
    """dialog used to add novel devices"""

    _box: QtWidgets.QComboBox
    _devices: List[str]
    _button_ok: QtWidgets.QPushButton
    _button_cancel: QtWidgets.QPushButton
    _button_all: QtWidgets.QPushButton
    _add_clicked: devices.Signal
    _all_clicked: devices.Signal

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
        self._add_clicked = devices.Signal()
        self._button_cancel.clicked.connect(self._cancel_selected)
        self._button_all.clicked.connect(self._all_selected)
        self._all_clicked = devices.Signal()

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
        policy_min = QtWidgets.QSizePolicy.Policy.Minimum
        lay = QtWidgets.QHBoxLayout()
        self._label = QtWidgets.QLabel(label)
        self._widgets = []
        for i, wdg in enumerate([self._label] + widgets):
            wdg.setFont(QtGui.QFont("Arial", 12))
            wdg.setSizePolicy(policy_min, policy_min)
            lay.addWidget(wdg)
            if i > 0:
                self._widgets += [wdg]

        # create the output widget
        super().__init__()
        self.setLayout(lay)
        self.setSizePolicy(policy_min, policy_min)
        self.setToolTip(tooltip)

    @property
    def widgets(self):
        return self._widgets

    @property
    def label(self):
        return self._label


class BigBrother(QtWidgets.QMainWindow):
    """
    BigBrother Application

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
    _container: QtWidgets.QGridLayout

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
        rec_path = join(ICON_PATH, "rec.png")
        rec_icon = QtGui.QPixmap(rec_path)
        rec_icon = rec_icon.scaled(ICON_SIZE, ICON_SIZE)
        self._rec_icon = QtGui.QIcon(rec_icon)

        stop_path = join(ICON_PATH, "stop.png")
        stop_icon = QtGui.QPixmap(stop_path)
        stop_icon = stop_icon.scaled(ICON_SIZE, ICON_SIZE)
        self._stop_icon = QtGui.QIcon(stop_icon)

        save_path = join(ICON_PATH, "save.png")
        save_icon = QtGui.QPixmap(save_path)
        save_icon = save_icon.scaled(ICON_SIZE, ICON_SIZE)
        save_icon = QtGui.QIcon(save_icon)

        add_path = join(ICON_PATH, "add.png")
        add_icon = QtGui.QPixmap(add_path)
        add_icon = add_icon.scaled(ICON_SIZE, ICON_SIZE)
        add_icon = QtGui.QIcon(add_icon)

        main_path = join(ICON_PATH, "main.png")
        main_icon = QtGui.QPixmap(main_path)
        main_icon = main_icon.scaled(ICON_SIZE, ICON_SIZE)
        main_icon = QtGui.QIcon(main_icon)

        # size policies
        policy_min = QtWidgets.QSizePolicy.Policy.Minimum
        policy_exp = QtWidgets.QSizePolicy.Policy.MinimumExpanding

        # container widget
        self._container = QtWidgets.QGridLayout()
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
        rec_obj.setLayout(rec_layout)
        rec_obj.setSizePolicy(policy_min, policy_min)

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
        opt_wdg = QtWidgets.QWidget()
        opt_wdg.setLayout(opt_layout)
        opt_wdg.setSizePolicy(policy_exp, policy_min)

        # setup the widget layout
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(cnt_widget)
        layout.addWidget(opt_wdg)
        central_widget = QtWidgets.QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        # window setup
        self._check_enabled()
        self.setWindowTitle("BigBrother")
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
        root = dirname(__file__)
        try:
            root = dirname(root)
        except Exception:
            pass
        return join(root, "_configuration.json")

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

        # apply
        self._save_wdg.setEnabled(~rec_checked & data_exist)
        self._frq_wdg.setEnabled(~rec_checked & cam_exist)
        self._rec_wdg.setEnabled(cam_exist)
        self._colormap_wdg.setEnabled(cam_exist)
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
                data = {i.strftime(devices.TIMESTAMP_FORMAT): v for i, v in buf}
                time = list(data.keys())[0].rsplit(".", 1)[0].replace(" ", "_")
                time = time.replace("-", "").replace(":", "")
                name = cam.device.id.replace("(", "").replace(")", "")
                name = name.replace(".", "").replace(" ", "_")
                name = name.replace(":", "").replace("-", "").lower()
                file = join(path, "-".join([time, name]) + ".npz")
                np.savez_compressed(file, **data)
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
        devs = devices.Device.get_available_devices()
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
            new_device = devices.LeptonDevice(id)
        elif id == "PI IMAGER":
            new_device = devices.OptrisPiDevice(id)
        else:
            new_device = devices.OpticalDevice(id)

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
        for row in np.arange(self._container.rowCount()):
            for col in np.arange(self._container.columnCount()):
                item = self._container.itemAtPosition(row, col)
                self._container.removeItem(item)

        # refresh the layout
        cams = list(self._buffer.keys())
        ncam = len(cams)
        if ncam > 0:
            rows = int(np.floor(ncam**0.5))
            cols = int(np.ceil(ncam / rows))
            for i, cam in enumerate(cams):
                row = int(np.floor(i // cols))
                col = i - row * cols
                self._container.addWidget(cam, row, col)
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
