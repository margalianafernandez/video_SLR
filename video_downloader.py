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


dataset = {
    TEST: dict(),
    TRAIN: dict(),
    VALIDATION: dict()
}


logging.basicConfig(filename='download_{}.log'.format(int(time.time())), filemode='w', level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


def trimme_file(inst):
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
    set = instance["split"]
    
    instance = {
        "video_id": instance["video_id"],
        "signer_id": instance["signer_id"]
    }
    
    if gloss not in dataset[set]:
        dataset[set][gloss] = []

    dataset[set][gloss] += [instance]


def convert_file_to_mp4(video_id, original_extension):
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
    user_agent = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'

    headers = {'User-Agent': user_agent,}
    
    if referer: headers['Referer'] = referer

    request = urllib.request.Request(url, None, headers)  # The assembled request

    response = urllib.request.urlopen(request)
    data = response.read()  # The data you need

    return data


def save_video(data, saveto):
    with open(saveto, 'wb+') as f:
        f.write(data)

    # please be nice to the host - take pauses and avoid spamming
    time.sleep(random.uniform(0.5, 1.5))


def download_youtube(url, dirname, video_id):
    raise NotImplementedError("Urllib cannot deal with YouTube links.")


def download_aslpro(url, dirname, video_id):
    saveto = os.path.join(dirname, '{}.swf'.format(video_id))
    if os.path.exists(saveto):
        logging.info('{} exists at {}'.format(video_id, saveto))
        return 

    data = request_video(url, referer='http://www.aslpro.com/cgi-bin/aslpro/aslpro.cgi')
    save_video(data, saveto)


def download_others(url, dirname, video_id):
    saveto = os.path.join(dirname, '{}.mp4'.format(video_id))
    if os.path.exists(saveto):
        logging.info('{} exists at {}'.format(video_id, saveto))
        return 
    
    data = request_video(url)
    save_video(data, saveto)


def select_download_method(url):
    if 'aslpro' in url:
        return download_aslpro
    elif 'youtube' in url or 'youtu.be' in url:
        return download_youtube
    else:
        return download_others


def check_youtube_dl_version():
    ver = os.popen('youtube-dl --version').read()

    assert ver, "youtube-dl cannot be found in PATH. Please verify your installation."
    assert ver >= '2020.03.08', "Please update youtube-dl to newest version."


def download_nonyt_videos():
    content = json.load(open(WLASL_FILE))

    if not os.path.exists(VIDEOS_FOLDER):
        os.mkdir(VIDEOS_FOLDER)

    for idx, entry in enumerate(content):

        if idx >= NUM_LABELS: break

        gloss = entry['gloss']
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
    content = json.load(open(WLASL_FILE))
    
    if not os.path.exists(VIDEOS_FOLDER):
        os.mkdir(VIDEOS_FOLDER)
    
    for idx, entry in enumerate(content):
        
        if idx >= NUM_LABELS: break
        
        gloss = entry['gloss']
        instances = entry['instances']

        for inst in instances:

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
                else:
                    logging.error('Unsuccessful downloading - youtube video url {}'.format(video_url))

                # please be nice to the host - take pauses and avoid spamming
                time.sleep(random.uniform(1.0, 1.5))
            

if __name__ == '__main__':

    logging.info('Start downloading non-youtube videos.')
    

    check_youtube_dl_version()
    logging.info('Start downloading youtube videos.')
    download_yt_videos()

    download_nonyt_videos()

    with open(DATASET_FILE, "w") as json_file:
        json.dump(dataset, json_file, indent=4)
