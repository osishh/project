import time
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from pdb import set_trace as st
from util import html


def main():
    # Parse options and force CPU mode
    opt = TestOptions().parse()
    opt.gpu_ids = []  # Force CPU mode
    opt.nThreads = 0  # ✅ Prevents multiprocessing issues in Windows
    opt.batchSize = 1  # Reduce memory usage
    opt.serial_batches = True  # No shuffle for testing
    opt.no_flip = True  # No flipping of images

    # Load dataset
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()

    # Create model and move it to CPU
    model = create_model(opt)
    model.netG_A.to("cpu")  # ✅ Moves generator to CPU
    visualizer = Visualizer(opt)

    # Create output directory
    web_dir = os.path.join("./ablation/", opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

    # Test
    print(f"Number of images in dataset: {len(dataset)}")
    if len(dataset) == 0:
        print("⚠️ No images found in dataset! Check your dataroot path.")

    for i, data in enumerate(dataset):
        model.set_input(data)
        visuals = model.predict()
        img_path = model.get_image_paths()
        print(f'Processing image {i + 1}/{len(dataset)}: {img_path}')
        visualizer.save_images(webpage, visuals, img_path)

    webpage.save()
    print("✅ Inference completed! Results saved in:", web_dir)


if __name__ == '__main__':
    main()
