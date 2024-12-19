import os
import shutil
import re
import PIL

from .dataclasses import Image


class FileManger(dict[str, Image]):
    def __init__(self, cache_path: str):
        self.cache_path = cache_path
        self.ORIGINAL_NAME = 'original'
        self.PREVIEW_NAME = 'preview'
        self.PREVIEW_FMT = 'webp'
        self.PREVIEW_MAX_SIZE = 1000
        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)
        super().__init__()

    def load_cached(self) -> list[str]:
        """Loads all the files of the cache folder into the dict

        Returns:
            list[str]: List of the keys of the inserted files
        """
        ids = []
        for directory in os.listdir(self.cache_path):
            dir_path = f'{self.cache_path}/{directory}'
            if not os.path.isdir(dir_path): continue
            name, counter = re.findall("(.*)_(\d)+$", directory)[0]
            orig_fmt, preview_fmt = '', ''
            for file in os.listdir(dir_path):
                file_name, fmt = re.findall("(.*)\.(.*)$", file)[0]
                if file_name == self.ORIGINAL_NAME:
                    orig_fmt = fmt
                elif file_name == self.PREVIEW_NAME:
                    preview_fmt = fmt
                else:
                    print(f'Found unexpected file {file} in directory {directory} in cache. TODO: Handle this')
            if orig_fmt != '' and preview_fmt != '' and directory not in self:
                self.__setitem__(directory, Image(
                    id=directory,
                    cache_path=self.cache_path,
                    name=name,
                    orig_file=self.ORIGINAL_NAME,
                    orig_fmt=orig_fmt,
                    preview_file=self.PREVIEW_NAME,
                    preview_fmt=preview_fmt))
                ids.append(directory)
        return ids

    def upload_image(self, old_path: str, filename: str) -> str:
        """Uploads the image and inserts it into the dictionary

        Args:
            old_path (str): Path the image should be loaded from
            name (str): name of the file with format (e.g. img.png)
        Returns:
            str: returns the key where the image can be found
        """
        name, fmt = re.findall("(.*)\.(.*)$", filename)[0]
        if not os.path.exists(self.cache_path):  # check if cache folder exists
            os.makedirs(self.cache_path)
        counter = 0
        new_folder = str(self.cache_path + name + "_{}")
        while os.path.isdir(new_folder.format(counter)):
            counter += 1
        id = f'{name}_{counter}'  # unique id for this image
        os.makedirs(new_folder.format(counter))
        new_path = f'{new_folder.format(counter)}/{self.ORIGINAL_NAME}.{fmt}'
        shutil.copy(old_path, new_path)
        self.__create_preview(new_path, f'{new_folder.format(counter)}/{self.PREVIEW_NAME}.{self.PREVIEW_FMT}')
        self.__setitem__(id, Image(
            id=id,
            cache_path=self.cache_path,
            name=name,
            orig_file=self.ORIGINAL_NAME,
            orig_fmt=fmt,
            preview_file=self.PREVIEW_NAME,
            preview_fmt=self.PREVIEW_FMT))
        return id

    def __create_preview(self, orig_path: str, preview_path: str):
        img = PIL.Image.open(orig_path)
        width, height = img.size
        if max(width, height) > self.PREVIEW_MAX_SIZE:
            scale = self.PREVIEW_MAX_SIZE / max(width, height)
            img = img.resize((int(width * scale), int(height * scale)))
        img.save(preview_path, optimize=True, quality=90)

    def export_image(self, id: str, export_path: str):
        img = self.__getitem__(id)
        if '.' not in export_path:
            export_path += '.' + img.orig_fmt
        shutil.copy(img.get_path_orig(), export_path)
        img.saved = True

    def __delitem__(self, id: str):
        img = self.__getitem__(id)
        dir = self.cache_path + img.id
        for file in os.listdir(dir):
            os.remove(dir + '/' + file)
        os.rmdir(dir)
        super().__delitem__(id)
