import os
from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import save_images
from util import html


if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.num_threads = 1   # test code only supports num_threads = 1
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.display_id = -1  # no visdom display
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    model = create_model(opt)
    model.setup(opt)
    # create website
    web_dir = opt.results_dir
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
    # test
    for i, data in enumerate(dataset):
        #img_path = model.get_image_paths()
        #print(img_path)
        # print(data.keys())
        # if i < 300:
        #     continue
        model.set_input(data)
        img_path = model.get_image_paths()
        if i % 300 == 0:
            print('{} / {}, img_path:'.format(i, len(dataset)), img_path[0])
        model.test()
        visuals = model.get_current_visuals()#in test the loadSize is set to the same as fineSize
        # print('visuals:', visuals)
        #if i % 5 == 0:
        #    print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(visuals, img_path, opt.results_dir)
        # exit()

    webpage.save()
