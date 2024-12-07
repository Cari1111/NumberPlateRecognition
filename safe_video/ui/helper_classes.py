import os
import shutil
import re
import PIL
import cv2

from .dataclasses import Image, Video, Media, FileVersion, FileVersionTemplate, ColorPalette

class FileManger(dict[str, Media]):
    def __init__(self, colors: ColorPalette):
        self.CACHE_PATH = "safe_video/upload_cache/"
        self.colors = colors
        self.ORIGINAL_TEMPLATE = FileVersionTemplate(name='original')
        self.PREVIEW_TEMPLATE_IMG = FileVersionTemplate(name='preview', fmt='webp', max_size=1000)
        self.PREVIEW_TEMPLATE_VID = FileVersionTemplate(name='preview', fmt='mp4', max_size=1000)
        self.ICON_TEMPLATE = FileVersionTemplate(name='icon', fmt='webp', max_size=100)
        self.IMAGE_FMTS = ['png', 'jpg']
        self.VIDEO_FMTS = ['mp4']
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
            orig_fmt, preview_fmt, icon_fmt = '', '', ''
            for file in os.listdir(dir_path):
                file_name, fmt = re.findall("(.*)\.(.*)$", file)[0]
                if file_name == self.ORIGINAL_TEMPLATE.name:
                    orig_fmt = fmt
                elif file_name == self.PREVIEW_TEMPLATE_IMG.name:
                    preview_fmt = fmt
                elif file_name == self.ICON_TEMPLATE.name:
                    icon_fmt = fmt
                else:
                    print(f'Found unexpected file {file} in directory {directory} in cache. TODO: Handle this')
            if orig_fmt != '' and preview_fmt != '' and directory not in self:
                media = Media(
                    id=directory,
                    cache_path=self.CACHE_PATH,
                    name=name,
                    orig_file=FileVersion(self.ORIGINAL_TEMPLATE.name, fmt=orig_fmt),
                    preview_file=FileVersion(self.PREVIEW_TEMPLATE_IMG.name, fmt=preview_fmt),
                    icon_file=FileVersion(self.ICON_TEMPLATE.name, fmt=icon_fmt))
                ids.append(directory)
                if orig_fmt in self.IMAGE_FMTS:
                    self.__setitem__(directory, Image(media))
                if orig_fmt in self.VIDEO_FMTS:
                    self.__setitem__(directory, Video(media, self.__get_aspect_ratio(media.get_path_orig())))
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
        if not os.path.exists(self.CACHE_PATH): # check if cache folder exists
            os.makedirs(self.CACHE_PATH)
        counter = 0
        new_folder = str(self.CACHE_PATH + name + "_{}")
        while os.path.isdir(new_folder.format(counter)):
            counter += 1
        os.makedirs(new_folder.format(counter))
        media = Media(
            id=f'{name}_{counter}', # unique id
            cache_path=self.CACHE_PATH,
            name=name,
            orig_file=FileVersion(self.ORIGINAL_TEMPLATE.name, fmt=fmt),
            preview_file=FileVersion(self.PREVIEW_TEMPLATE_IMG.name, fmt=self.PREVIEW_TEMPLATE_IMG.fmt),
            icon_file=FileVersion(self.ICON_TEMPLATE.name, fmt=self.ICON_TEMPLATE.fmt))
        shutil.copy(old_path, media.get_path_orig())
        if fmt in self.IMAGE_FMTS:
            self.__create_preview_and_icon_from_image(orig_path=media.get_path_orig(), preview_path=media.get_path_preview(), icon_path=media.get_path_icon())
            self.__setitem__(media.id, Image(media))
        elif fmt in self.VIDEO_FMTS:
            media.preview_file.fmt = media.orig_file.fmt # TODO: change this
            self.__create_preview_and_icon_from_video(orig_path=media.get_path_orig(), preview_path=media.get_path_preview(), icon_path=media.get_path_icon())
            self.__setitem__(media.id, Video(media, self.__get_aspect_ratio(media.get_path_orig())))
        return media.id


    def __create_preview_and_icon_from_image(self, orig_path: str, preview_path: str, icon_path: str):
        img = PIL.Image.open(orig_path)
        width, height = img.size
        if max(width, height) > self.PREVIEW_TEMPLATE_IMG.max_size:
            scale = self.PREVIEW_TEMPLATE_IMG.max_size/max(width, height)
            img = img.resize((int(width*scale), int(height*scale)))
        img.save(preview_path, optimize=True, quality=90)
        if max(width, height) > self.ICON_TEMPLATE.max_size:
            scale = self.ICON_TEMPLATE.max_size/max(width, height)
            img = img.resize((int(width*scale), int(height*scale)))
        img.save(icon_path, optimize=True, quality=90)

    def __create_preview_and_icon_from_video(self, orig_path: str, preview_path: str, icon_path: str):
        shutil.copy(orig_path, preview_path)
        video = cv2.VideoCapture(orig_path)
        _, image = video.read()
        cv2.imwrite(icon_path, image)
        img = PIL.Image.open(icon_path)
        width, height = img.size
        if max(width, height) > self.PREVIEW_TEMPLATE_IMG.max_size:
            scale = self.PREVIEW_TEMPLATE_IMG.max_size/max(width, height)
            img = img.resize((int(width*scale), int(height*scale)))
        img.save(icon_path, optimize=True, quality=90)

    def __get_aspect_ratio(self, path):
        video = cv2.VideoCapture(path)
        _, image = video.read()
        height, width, _ = image.shape
        return(width/height)


    def export_image(self, id: str, export_path: str):
        img = self.__getitem__(id)
        if '.' not in export_path:
            export_path += '.' + img.orig_fmt
        shutil.copy(img.get_path_orig(), export_path)
        img.saved = True

    def __delitem__(self, id: str):
        img = self.__getitem__(id)
        dir = self.CACHE_PATH + img.id
        for file in os.listdir(dir):
            os.remove(dir + '/' + file)
        os.rmdir(dir)
        super().__delitem__(id)