import os
import glob
import shutil

src = '/media/efklidis/4TB/medical-annotations/train/'

dst = '/media/efklidis/4TB/medical-annotations-reformatted/train/'

seqs = glob.glob(os.path.join(src, "*"))

for seq in seqs:
    # make path
    new_seq = seq.replace("annotations", "annotations-reformatted")
    os.makedirs(new_seq, exist_ok=True)

    input_path = os.path.join(new_seq, "input")
    annotator_one_path = os.path.join(new_seq, "annotator_one")
    annotator_two_path = os.path.join(new_seq, "annotator_two")
    annotator_three_path = os.path.join(new_seq, "annotator_three")

    os.makedirs(input_path, exist_ok=True)
    os.makedirs(annotator_one_path, exist_ok=True)
    os.makedirs(annotator_two_path, exist_ok=True)
    os.makedirs(annotator_three_path, exist_ok=True)


    frames = [x.split("/")[-1] for x in glob.glob(os.path.join(seq, "*"))]

    [os.makedirs(x, exist_ok=True) for x in [annotator_one_path + '/' + x for x in frames]]
    [os.makedirs(x, exist_ok=True) for x in [annotator_two_path + '/' + x for x in frames]]
    [os.makedirs(x, exist_ok=True) for x in [annotator_three_path + '/' + x for x in frames]]
    [os.makedirs(x, exist_ok=True) for x in [input_path + '/' + x for x in frames]]

    print(frames)

    full_frames_input = [x for x in glob.glob(os.path.join(new_seq, "input", "*"))]

    for frame in full_frames_input:

        seq_index, folder, frame_no =frame.split("/")[-3:]
        old_input = os.path.join(src, seq_index, frame_no, frame_no + ".png")
        old_segment_one = os.path.join(seq, frame_no, "segmentation", "25d0ad68-4480-47b3-adef-22fe1d762415.png")
        old_segment_two = os.path.join(seq, frame_no, "segmentation", "046d461d-f02e-4de8-9c0c-f5e3be63206e.png")
        old_segment_three = os.path.join(seq, frame_no, "segmentation", "2516426f-f491-4281-a767-5bc6d463a44f.png")

        # move segmentation and mask
        shutil.copyfile(old_input, frame + "/input.png")

        shutil.copyfile(old_segment_one, frame.replace("input", "annotator_one") + "/segmentation.png")
        shutil.copyfile(old_segment_two, frame.replace("input", "annotator_two") + "/segmentation.png")
        shutil.copyfile(old_segment_three, frame.replace("input", "annotator_three") + "/segmentation.png")






