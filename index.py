import shutil
import threading

import cv2
import wx
import extract_and_save_features as easf
import find_image_in_video as fiiv
from moviepy.editor import *


class MyFrame(wx.Frame):
    def __init__(self):
        super().__init__(parent=None, title='Nhận diện đối tượng trong video', pos=(500, 500), size=(800, 600))
        panel = wx.Panel(self)
        my_sizer = wx.BoxSizer(wx.VERTICAL)
        #
        self.find_process_label = wx.StaticText(panel, style=wx.ALIGN_CENTER_HORIZONTAL, size=(400, 40))
        self.find_process_label.SetLabel("Bạn có thể bắt đầu import video vào db")
        my_sizer.Add(self.find_process_label, 0, wx.ALL | wx.CENTER, 5)
        #
        my_btn = wx.Button(panel, label='Import video to db')
        my_btn.Bind(wx.EVT_BUTTON, self.on_choose_video)
        my_sizer.Add(my_btn, 0, wx.ALL | wx.CENTER, 5)
        #
        self.find_process_label = wx.StaticText(panel, style=wx.ALIGN_CENTER_HORIZONTAL, size=(400, 40))
        self.find_process_label.SetLabel("Bạn có thể chọn 1 ảnh để bắt đầu tìm kiếm trong db")
        my_sizer.Add(self.find_process_label, 0, wx.CENTER, 5)
        #
        my_btn = wx.Button(panel, label='Chọn ảnh để tìm kiếm')
        my_btn.Bind(wx.EVT_BUTTON, self.on_choose_img)
        my_sizer.Add(my_btn, 0, wx.ALL | wx.CENTER, 5)
        #
        self.log_box = wx.TextCtrl(panel, size=(400, 200), style=wx.TE_MULTILINE)
        my_sizer.Add(self.log_box, 0, wx.ALL | wx.CENTER, 5)
        #
        #
        panel.SetSizer(my_sizer)
        self.Show()

        #
        self.result = {}

    def on_choose_video(self, event):
        openFileDialog = wx.FileDialog(frame, "Open", "", "", "Video files (*.mp4)|*.mp4",
                                       wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)
        openFileDialog.ShowModal()
        path = openFileDialog.GetPath()
        openFileDialog.Destroy()

        if path:
            x = threading.Thread(target=self.handle_import, args=(path, self.find_process_label))
            x.start()

    def on_choose_img(self, event):
        openFileDialog = wx.FileDialog(frame, "Open", "", "", "Images files (*.png)|*.png|*.jpg",
                                       wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)
        openFileDialog.ShowModal()
        path = openFileDialog.GetPath()
        openFileDialog.Destroy()

        if path:
            img = cv2.imread(path)
            x = threading.Thread(target=self.handle_find, args=(img, self.find_process_label))
            x.start()

    def on_found(self, name, second, percentage):
        # self.log_box.AppendText(name + " " + str(second) + "s " + str(percentage) + " %\n")
        # self.log_box.Refresh()
        p = self.result.get(name, [])
        p.append(second)
        self.result[name] = p

    def on_finished(self):
        for key in self.result:
            name = key
            time_range = sorted(self.result.get(key))
            mi = min(time_range)
            ma = max(time_range)
            self.log_box.AppendText(name + " " + str(mi) + " --> " + str(ma) + " \n")
            if ma > mi:
                cut_video("input/" + name, mi, ma)
            else:
                cut_video("input/" + name, mi, ma + 1)

    def handle_find(self, img, label):
        self.result = {}
        label.SetLabel("Finding ............")
        fiiv.find(img, self.on_found, self.on_finished)
        label.SetLabel("Find process has ended.")

    def handle_import(self, path, text_ctrl):
        # copy to input folder
        text = "Processing " + path + " ......"
        print(text)
        text_ctrl.SetLabel(text)

        file = shutil.copy(path, "./input/")
        video = cv2.VideoCapture(file)
        videoName = easf.get_file_name_from_path(file)
        # extract features
        easf.extract_and_save(videoName, video)
        text = path + " is processed"
        print(text)
        text_ctrl.SetLabel(text)


def cut_video(path, f, t):
    print(path)
    clip = (VideoFileClip(path).subclip(f, t))
    name = str(f) + str(t)
    clip.write_videofile("output/output-%s.mp4" % name)


def init_output_folder():
    path = 'output/'
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except OSError:
        print("Error: Creating directory of " + path)


def init_input_folder():
    path = 'input/'
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except OSError:
        print("Error: Creating directory of " + path)


if __name__ == '__main__':
    init_output_folder()
    init_input_folder()
    app = wx.App()
    frame = MyFrame()
    app.MainLoop()
