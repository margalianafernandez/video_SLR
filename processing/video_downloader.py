import os
import sys
import json
import time
import ffmpeg
import random
import logging
import subprocess
import urllib.request
from data_constants import *
from os.path import join


# Dataset dictionary to store downloaded video instances
dataset = {
    TEST: dict(),
    TRAIN: dict(),
    VALIDATION: dict()
}


# Configure logging to both file and stdout
logging.basicConfig(filename='download_{}.log'.format(int(time.time())), filemode='w', level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


def trimme_file(inst):
    """
    Trim video file based on frame range.

    Args:
        inst (dict): Video instance data.

    Returns:
        None
    """
    video_id = inst["video_id"]
    frame_start = inst["frame_start"]
    frame_end = inst["frame_end"]
    fps = inst["fps"]

    valid_trim = frame_start and  frame_end and  frame_start < frame_end
    if not valid_trim: return
    
    start_time = frame_start / fps
    end_time = frame_end / fps
    
    input_file = os.path.join(VIDEOS_FOLDER, video_id + '.mp4')
    output_file = os.path.join(VIDEOS_FOLDER, video_id + '_tmp.mp4')

    cmd = [
        'ffmpeg',
        '-i', input_file,
        '-ss', str(start_time),
        '-to', str(end_time),
        '-c', 'copy',
        output_file
    ]

    # Run the ffmpeg command using subprocess
    rv = subprocess.run(cmd)

    if rv.returncode == 0:
        os.remove(input_file)
        os.rename(output_file, input_file)
    else:
        print(f"Error while trimming video - video {video_id}")


def register_video_downloaded(gloss, instance):
    """
    Register downloaded video instance in the dataset.

    Args:
        gloss (str): Gloss associated with the video instance.
        instance (dict): Video instance data.

    Returns:
        None
    """
    set = instance["split"]
    
    instance = {
        "video_id": instance["video_id"],
        "signer_id": instance["signer_id"]
    }
    
    if gloss not in dataset[set]:
        dataset[set][gloss] = []

    dataset[set][gloss] += [instance]


def convert_file_to_mp4(video_id, original_extension):
    """
    Convert non-MP4 file to MP4 format.

    Args:
        video_id (str): Video ID.
        original_extension (str): Original file extension.

    Returns:
        None
    """
    input_file = join(VIDEOS_FOLDER, video_id+original_extension)
  
    # Create the output file path with the '.mp4' extension
    output_file = join(VIDEOS_FOLDER, os.path.splitext(os.path.basename(input_file))[0] + '.mp4')

    # Define the input stream (SWF) and output stream (MP4)
    input_stream = ffmpeg.input(input_file)
    output_stream = ffmpeg.output(input_stream, output_file)

    # Run the conversion using ffmpeg-python
    ffmpeg.run(output_stream)

    os.remove(input_file)


def request_video(url, referer=''):
    """
    Make a request to a URL and retrieve video data.

    Args:
        url (str): URL to request.
        referer (str): Referer URL.

    Returns:
        bytes: Retrieved video data.
    """
    user_agent = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'

    headers = {'User-Agent': user_agent,}
    
    if referer: headers['Referer'] = referer

    request = urllib.request.Request(url, None, headers)  # The assembled request

    response = urllib.request.urlopen(request)
    data = response.read()  # The data you need

    return data


def save_video(data, saveto):
    """
    Save video data to a file.

    Args:
        data (bytes): Video data to save.
        saveto (str): File path to save the video data.

    Returns:
        None
    """
    with open(saveto, 'wb+') as f:
        f.write(data)

    # please be nice to the host - take pauses and avoid spamming
    time.sleep(random.uniform(0.5, 1.5))


def download_youtube(url, dirname, video_id):
    """
    Placeholder function for downloading YouTube videos.

    Args:
        url (str): YouTube video URL.
        dirname (str): Directory name.
        video_id (str): Video ID.

    Raises:
        NotImplementedError: Placeholder function, YouTube download not supported.
    """
    raise NotImplementedError("Urllib cannot deal with YouTube links.")

def download_aslpro(url, dirname, video_id):
    """
    Download videos from ASLPro website and save as .swf file.

    Args:
        url (str): URL of the video to download.
        dirname (str): Directory to save the downloaded video.
        video_id (str): ID of the video.

    Returns:
        None
    """
    saveto = os.path.join(dirname, '{}.swf'.format(video_id))
    if os.path.exists(saveto):
        logging.info('{} exists at {}'.format(video_id, saveto))
        return 

    data = request_video(url, referer='http://www.aslpro.com/cgi-bin/aslpro/aslpro.cgi')
    save_video(data, saveto)


def download_others(url, dirname, video_id):
    """
    Download videos from other sources and save as .mp4 file.

    Args:
        url (str): URL of the video to download.
        dirname (str): Directory to save the downloaded video.
        video_id (str): ID of the video.

    Returns:
        None
    """
    saveto = os.path.join(dirname, '{}.mp4'.format(video_id))
    if os.path.exists(saveto):
        logging.info('{} exists at {}'.format(video_id, saveto))
        return 
    
    data = request_video(url)
    save_video(data, saveto)


def select_download_method(url):
    """
    Select the appropriate download method based on the URL.

    Args:
        url (str): URL of the video to download.

    Returns:
        function: The appropriate download function.
    """
    if 'aslpro' in url:
        return download_aslpro
    elif 'youtube' in url or 'youtu.be' in url:
        return download_youtube
    else:
        return download_others


def check_youtube_dl_version():
    """
    Check if youtube-dl is installed and its version is up to date.

    Args:
        None

    Returns:
        None
    """
    ver = os.popen('youtube-dl --version').read()

    assert ver, "youtube-dl cannot be found in PATH. Please verify your installation."
    assert ver >= '2020.03.08', "Please update youtube-dl to the newest version."


def download_nonyt_videos():
    """
    Download non-YouTube videos from the provided URLs.

    Args:
        None

    Returns:
        None
    """
    content = json.load(open(WLASL_FILE))

    if not os.path.exists(VIDEOS_FOLDER):
        os.mkdir(VIDEOS_FOLDER)

    for entry in content:
        gloss = entry['gloss']

        if gloss not in LABELS: 
            continue 

        instances = entry['instances']

        for inst in instances:
            video_url = inst['url']
            video_id = inst['video_id']
            
            logging.info('gloss: {}, video: {}.'.format(gloss, video_id))

            download_method = select_download_method(video_url)    
            
            if download_method == download_youtube:
                logging.warning('Skipping YouTube video {}'.format(video_id))
                continue

            try:
                download_method(video_url, VIDEOS_FOLDER, video_id)

                if download_method == download_aslpro:
                    convert_file_to_mp4(video_id, ".swf")

                trimme_file(inst)
                register_video_downloaded(gloss, inst)

            except Exception as e:
                logging.error('Unsuccessful downloading - video {}'.format(video_id))


def download_yt_videos():
    """
    Download YouTube videos from the provided URLs.

    Args:
        None

    Returns:
        None
    """
    content = json.load(open(WLASL_FILE))
    
    if not os.path.exists(VIDEOS_FOLDER):
        os.mkdir(VIDEOS_FOLDER)
    
    for entry in content:
        gloss = entry['gloss']

        if gloss not in LABELS: 
            continue        
        
        instances = entry['instances']
        video_downloaded = 0

        for inst in instances:

            if video_downloaded >= MAX_SAMPLES_LABEL: 
                break

            if video_downloaded >= MIN_SAMPLES_LABEL and inst['end_time'] > 6: 
                continue
            
            video_url = inst['url']
            video_id = inst['video_id']

            if 'youtube' not in video_url and 'youtu.be' not in video_url:
                continue

            if os.path.exists(os.path.join(VIDEOS_FOLDER, video_url[-11:] + '.mp4')) or os.path.exists(os.path.join(VIDEOS_FOLDER, video_url[-11:] + '.mkv')):
                logging.info('YouTube videos {} already exists.'.format(video_url))
                continue
            
            else:
                cmd = "youtube-dl \"{}\" -o \"{}{}.%(ext)s\""
                cmd = cmd.format(video_url, VIDEOS_FOLDER + os.path.sep, video_id)

                rv = os.system(cmd)

                if not rv:
                    
                    if os.path.exists(join(VIDEOS_FOLDER, video_id+".mkv")):
                        convert_file_to_mp4(video_id, ".mkv")

                    trimme_file(inst)
                    register_video_downloaded(gloss, inst)
                    video_downloaded += 1
                else:
                    logging.error('Unsuccessful downloading - youtube video url {}'.format(video_url))

                # please be nice to the host - take pauses and avoid spamming
                time.sleep(random.uniform(1.0, 1.5))


def remove_unwanted_files():
    # Iterate through the files and remove those that are not .mp4
    files = os.listdir(VIDEOS_FOLDER)

    for file in files:
        if not file.endswith('.mp4'):
            file_path = os.path.join(VIDEOS_FOLDER, file)
            os.remove(file_path)


if __name__ == '__main__':

    logging.info('Start downloading non-youtube videos.')
    

    check_youtube_dl_version()
    logging.info('Start downloading youtube videos.')
    download_yt_videos()

    download_nonyt_videos()
    
    with open(DATASET_FILE, "w") as json_file:
        json.dump(dataset, json_file, indent=4)

    remove_unwanted_files()
