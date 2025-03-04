import os
import shutil
import re
import PIL
import warnings
import cv2
import pickle
import numpy as np
import base64
import io
import shutil
import flet as ft
import asyncio

from .dataclasses import Image, Video, Media, FileVersion, FileVersionTemplate, ColorPalette, Version
from .components import ModelTile
from safe_video.number_plate_recognition import ObjectDetection, Censor, apply_censorship, merge_results_list, save_video_with_status, StopAsyncGenerator
from ultralytics.engine.results import Results


class ModelManager():
    def __init__(self, bounding_box_func):
        self.detection: ObjectDetection = ObjectDetection()
        self.cls_file_path = 'safe_video/upload_cache/cls_file.pkl'
        self.cls = {}
        if os.path.isfile(self.cls_file_path):
            with open(self.cls_file_path, 'rb') as file:
                self.cls = pickle.load(file)
        self.active: dict[str, bool] = {c: True for c in self.cls.keys()}
        self.results: dict[str, dict[str, Results]] = dict()  # dict[cls_id][img_id]
        self.bounding_box_func = bounding_box_func

    def toggle_active(self, cls_id):
        self.active[cls_id] = not self.active[cls_id]

    def get_possible_cls(self):
        return self.detection.get_classes()

    def insert_new_cls(self, name, classes, active=True):
        counter = 1
        id = name
        while id in self.cls.keys():
            id = name + ' ' + str(counter)
            counter += 1
        self.cls[id] = classes
        self.active[id] = active
        self.update_cls_file()
        return id

    def edit_cls(self, old_name, new_name, classes):
        if old_name in self.results:
            del self.results[old_name]
        if new_name == old_name:
            self.cls[old_name] = classes
            self.update_cls_file()
            return old_name
        else:
            del self.cls[old_name]
            active = self.active[old_name]
            del self.active[old_name]
            return self.insert_new_cls(new_name, classes, active)

    def delete_cls(self, id):
        if id in self.results:
            del self.results[id]
        del self.cls[id]
        del self.active[id]
        self.update_cls_file()

    def update_cls_file(self):
        with open(self.cls_file_path, 'wb') as file:
            pickle.dump(self.cls, file, protocol=pickle.HIGHEST_PROTOCOL)

    def get_bounding_box_image(self, cls_id: str, img: Image):
        bb_image = merge_results_list(self.analyze_or_from_cache(cls_id, img)).plot()
        image = PIL.Image.fromarray(bb_image)
        buffered = io.BytesIO()
        image.save(buffered, format="jpeg")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def get_blurred_image(self, cls_ids: str, img: Image, options) -> np.ndarray:
        img_loaded = cv2.imread(img.get_path(Version.ORIG))[:, :, ::-1]
        for cls_id in cls_ids:
            img_loaded = apply_censorship(img_loaded, self.analyze_or_from_cache(cls_id, img)[-1], **options[cls_id])
        return img_loaded

    def get_analyzed_video(self, cls_ids: list[str], video: Video, page: ft.Page, pb: ft.ProgressBar, options) -> list[Results]:
        if not video.censored_available:
            shutil.copy(video.get_path(Version.ORIG), video.get_path(Version.PREVIEW_CENSORED))

        for cls_id in cls_ids:
            video_results: list[Results] = self.analyze_or_from_cache(cls_id, video, page, pb, options)
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(
                self.do_video(page, pb, cls_id, "Saving", save_video_with_status, 3,
                                results=self.results[cls_id][video.id], output_path=video.get_path(Version.PREVIEW_CENSORED),
                                original_video_path=video.get_path(Version.PREVIEW_CENSORED),
                                class_filter=self.cls[cls_id][-1], **options[cls_id]))

        video.censored_available = True

        shutil.copy(video.get_path(Version.PREVIEW_CENSORED), video.get_path(Version.ORIG_CENSORED))
        return video_results

    async def do_video(self, page: ft.Page, pb: ft.ProgressBar, cls_id: str, doing: str, fun, non_kw_args: int, **kwargs):
        pb_text = pb.controls[0]
        progress_value = pb.controls[1]
        pb_text.value = f"{doing} video, {cls_id}"
        page.update()
        status = fun(*list(kwargs.values())[:non_kw_args], **dict(list(kwargs.items())[non_kw_args:]))
        last_update_time = asyncio.get_event_loop().time()
        async for progress in status:
            if not isinstance(progress, (float, int)):
                raise StopAsyncGenerator(progress)
            current_time = asyncio.get_event_loop().time()
            if current_time - last_update_time >= 0.1:
                progress_value.value = progress
                page.update()
                last_update_time = current_time

    def analyze_or_from_cache(self, cls_id, media: Media, page: ft.Page = None, pb: ft.ProgressBar = None, options=None) -> list[Results]:
        if cls_id not in self.results:
            self.results[cls_id] = dict()
        if media.id not in self.results[cls_id] and type(media) is Image:
            img_loaded = cv2.imread(media.get_path(Version.ORIG))
            img_loaded = img_loaded[:, :, ::-1]
            self.results[cls_id][media.id] = self.detection.process_image(
                img_loaded, self.cls[cls_id], conf_thresh=0.25)
        if media.id not in self.results[cls_id] and type(media) is Video:
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(
                    self.do_video(page, pb, cls_id, "Processing", self.detection.process_video_with_status, 2,
                                  video_path=media.get_path(Version.PREVIEW_CENSORED), classes=self.cls[cls_id],
                                  conf_thresh=0.25, verbose=False))
                while loop.is_running(): pass
            except StopAsyncGenerator as v:
                self.results[cls_id][media.id] = v.value
        return self.results[cls_id][media.id]


class FileManger(dict[str, Media]):
    def __init__(self, colors: ColorPalette):
        self.CACHE_PATH = "safe_video/upload_cache/"
        self.colors = colors
        self.PREVIEW_MAX_SIZE = 1000
        self.ICON_MAX_SIZE = 100
        self.IMAGE_FMTS = ['png', 'jpg', 'jpeg']
        self.VIDEO_FMTS = ['mp4']
        self.blur_orig = True
        if not os.path.exists(self.CACHE_PATH):
            os.makedirs(self.CACHE_PATH)
        super().__init__()

    def load_cached(self) -> list[str]:
        """Loads all the files of the cache folder into the dict

        Returns:
            list[str]: List of the keys of the inserted files
        """
        ids = []
        for directory in os.listdir(self.CACHE_PATH):
            dir_path = f'{self.CACHE_PATH}/{directory}'
            if not os.path.isdir(dir_path): continue
            name, counter = re.findall("(.*)_(\d)+$", directory)[0]
            media = Media(id=directory, cache_path=self.CACHE_PATH, name=name)
            censored_files = 0
            for file in os.listdir(dir_path):
                file_name, fmt = re.findall("(.*)\.(.*)$", file)[0]
                for version in Version:
                    if file_name == media.files[version].name:
                        media.files[version].fmt = fmt
                        if version is Version.ORIG_CENSORED or version is Version.PREVIEW_CENSORED:
                            censored_files += 1
            if censored_files == 2:
                media.censored_available = True
                # TODO: Check if all files are there and no other files are in the directory
            media.set_orig_fmt(media.files[Version.ORIG].fmt)
            if media.files[Version.ORIG].fmt is not None and directory not in self:
                if media.files[Version.ORIG].fmt in self.IMAGE_FMTS:
                    self.__setitem__(directory, Image(media))
                elif media.files[Version.ORIG].fmt in self.VIDEO_FMTS:
                    self.__setitem__(directory, Video(media, self.__get_aspect_ratio(media.get_path(Version.ORIG))))
                else: continue
                ids.append(directory)
            else:
                print(f'Found directory {directory} with not enough files to be valid in cache. TODO: Handle this')
        return ids

    def upload_media(self, old_path: str, filename: str) -> str:
        """Uploads the image and inserts it into the dictionary

        Args:
            old_path (str): Path the image should be loaded from
            name (str): name of the file with format (e.g. img.png)
        Returns:
            str: returns the key where the image can be found
        """
        name, fmt = re.findall("(.*)\.(.*)$", filename)[0]
        if not os.path.exists(self.CACHE_PATH):  # check if cache folder exists
            os.makedirs(self.CACHE_PATH)
        counter = 0
        new_folder = str(self.CACHE_PATH + name + "_{}")
        while os.path.isdir(new_folder.format(counter)):
            counter += 1
        os.makedirs(new_folder.format(counter))
        media = Media(id=f'{name}_{counter}', cache_path=self.CACHE_PATH, name=name)
        media.set_orig_fmt(fmt)
        shutil.copy(old_path, media.get_path(Version.ORIG))
        if fmt in self.IMAGE_FMTS:
            self.__create_new_version_of_image(media.get_path(
                Version.ORIG), media.get_path(Version.PREVIEW), self.PREVIEW_MAX_SIZE)
            self.__create_new_version_of_image(media.get_path(
                Version.ORIG), media.get_path(Version.ICON), self.ICON_MAX_SIZE)
            self.__setitem__(media.id, Image(media))
        elif fmt in self.VIDEO_FMTS:
            # media.preview_file.fmt = media.orig_file.fmt # TODO: change this
            self.__create_preview_and_icon_from_video(orig_path=media.get_path(
                Version.ORIG), preview_path=media.get_path(Version.PREVIEW), icon_path=media.get_path(Version.ICON))
            self.__setitem__(media.id, Video(media, self.__get_aspect_ratio(media.get_path(Version.ORIG))))
        return media.id

    def __create_new_version_of_image(self, orig_path: str, new_path: str, max_size: int):
        img = PIL.Image.open(orig_path)
        width, height = img.size
        if max(width, height) > max_size:
            scale = max_size / max(width, height)
            img = img.resize((int(width * scale), int(height * scale)))
        img.save(new_path, optimize=True, quality=90)

    def create_blurred_imgs(self, id, blur_result):
        img = self.__getitem__(id)
        censored_img = PIL.Image.fromarray(blur_result)
        censored_img.save(img.get_path(Version.ORIG_CENSORED))
        self.__create_new_version_of_image(img.get_path(Version.ORIG_CENSORED),
                                           img.get_path(Version.PREVIEW_CENSORED), self.PREVIEW_MAX_SIZE)
        img.censored_available = True

    def __create_preview_and_icon_from_video(self, orig_path: str, preview_path: str, icon_path: str):
        shutil.copy(orig_path, preview_path)
        video = cv2.VideoCapture(orig_path)
        _, image = video.read()
        cv2.imwrite(icon_path, image)
        img = PIL.Image.open(icon_path)
        width, height = img.size
        if max(width, height) > self.PREVIEW_MAX_SIZE:
            scale = self.PREVIEW_MAX_SIZE / max(width, height)
            img = img.resize((int(width * scale), int(height * scale)))
        img.save(icon_path, optimize=True, quality=90)

    def __get_aspect_ratio(self, path):
        video = cv2.VideoCapture(path)
        _, image = video.read()
        height, width, _ = image.shape
        return (width / height)

    def toggle_blur_orig(self):
        self.blur_orig = not self.blur_orig

    def export_image(self, id: str, export_path: str):
        img = self.__getitem__(id)
        newest_version = Version.ORIG if img.files[Version.ORIG_CENSORED] is None else Version.ORIG_CENSORED
        if '.' not in export_path:
            export_path += '.' + img.files[newest_version].fmt
        shutil.copy(img.get_path(newest_version), export_path)
        img.saved = True

    def __delitem__(self, id: str):
        img = self.__getitem__(id)
        dir = self.CACHE_PATH + img.id
        for file in os.listdir(dir):
            os.remove(dir + '/' + file)
        os.rmdir(dir)
        super().__delitem__(id)
