from google_drive_downloader import GoogleDriveDownloader as gdd
import os


class Downloader:
    @staticmethod
    def download(google_id, output_path):
        if not os.path.exists(output_path):
            gdd.download_file_from_google_drive(file_id=google_id,
                                                showsize=True,
                                                dest_path=output_path,
                                                unzip=True)
        return output_path
