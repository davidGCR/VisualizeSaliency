from dinamycImage import *
import cv2, os
import argparse

def fromFrames(path, nDynamicImages, debugg_mode=False):
  frames_list = os.listdir(path)
  frames_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
  total_frames = len(frames_list)

  seqLen = int(total_frames/nDynamicImages)

  sequences = [
      frames_list[x : x + seqLen] for x in range(0, total_frames, seqLen)
  ]
  if debugg_mode:
      print('seqLen: ',seqLen)
      print('n sequences: ', len(sequences))
  img = None   
  for seq in sequences:
    if len(seq) == seqLen:
      frames = []
      for frame in seq:
        img_dir = str(path) + "/" + frame
        img = Image.open(img_dir).convert("RGB")
        img = np.array(img)
        frames.append(img)
      if debugg_mode:
        print('->total frames for Di: ', len(frames))
      imgPIL, img = getDynamicImage(frames)
      cv2.imshow('Di', img)
      while True:
        if cv2.waitKey(70) & 0xFF == ord('q'):
          break   

def fromVideo(path, seq_duration):
  cap = cv2.VideoCapture(path)
  video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  fps = cap.get(cv2.CAP_PROP_FPS)
  duration = video_length / fps
  print('fps:',fps,'video duration:',duration)
  count = 0
  frames = []
  current_time = 0
  while count < video_length - 1:
  # while current_time < duration:
    current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
    print('current_time ',current_time, 'seq_duration ',seq_duration)
    
    if current_time > seq_duration:
      print('========================\t num frames:', len(frames))
      diPIL, di = getDynamicImage(frames)
      cv2.imshow('Di', di)
      frames = []
      seq_duration = current_time + seq_duration
      
    success, image = cap.read()
    if success:
        frames.append(image)
        cv2.imshow('Frame', image)
        if cv2.waitKey(70) & 0xFF == ord('q'):
          break
    count += 1
  
  if len(frames) > 0:
    diPIL, di = getDynamicImage(frames)
    cv2.imshow('Di', di)
    cv2.waitKey(70)

def __main__():
  parser = argparse.ArgumentParser()
  parser.add_argument("--source_type", type=str, default="video")
  parser.add_argument("--video_path", type=str, default="/media/david/datos/Violence DATA/HockeyFights/videos/violence/fi1_xvid.avi", help="Directory containing video")
  parser.add_argument("--num_dimages", type=int, default=10)
  parser.add_argument("--seq_duration", type=float, default=1)
  parser.add_argument("--debug_mode", type=bool, default=False)
  args = parser.parse_args()
  source_type = args.source_type
  vid_name = args.video_path
  seq_duration = args.seq_duration
  num_dimages = args.num_dimages
  debug_mode = args.debug_mode
  
  if source_type == "video":
    fromVideo(vid_name, seq_duration)
  elif source_type == "frames":
    fromFrames(vid_name, num_dimages,debug_mode)

__main__()