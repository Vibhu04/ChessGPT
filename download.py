import wget
from zipfile import ZipFile
import argparse
import os


def main():

    parser = arg_parser()
    args, unknown = parser.parse_known_args()

    site_url = "https://archive.org/download/KingBase2018/"
    file_name = "KingBase2018-pgn.zip"
    out_folder = args.download_location

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    chess_archive_url = site_url + file_name

    print("Downloading chess archive, it might take a few moments...")
    wget.download(chess_archive_url, out = out_folder)
    print("\nFinished downloading.")

    extract_folder = args.extract_location
    if not os.path.exists(extract_folder):
        os.makedirs(extract_folder)
    
    if out_folder[-1] != '/':
        out_folder += '/'

    print("Extracting...")        
    with ZipFile(out_folder + file_name, 'r') as zObject:
        zObject.extractall(path=extract_folder)

    print("Files extracted, stored at", extract_folder)


def arg_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument('--download_location', type=str, default='data/', help="Location for downloading the chess archive.")
    parser.add_argument('--extract_location', type=str, default='data/', help="Location for extracting the downloaded files.")

    return parser



if __name__ == "__main__":

    main()






