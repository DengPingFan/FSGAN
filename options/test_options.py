from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        parser.add_argument('--results_dir', type=str, default='./ours_I2S_results/', help='saves results here.')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        parser.add_argument('--which_epoch', type=str, default='100', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--how_many', type=int, default=2000, help='how many test images to run')
        parser.add_argument('--save2', action='store_true', help='only save real_A and fake_B')
        parser.add_argument('--lm_dir', type=str, default='/home/pz1/datasets/fss/FS2K_data/test/landmark/', help='path to facial landmarks')
        parser.add_argument('--bg_dir', type=str, default='/home/pz1/datasets/fss/FS2K_data/test/mask/', help='path to background masks')
        parser.add_argument('--data_json', default='/home/pz1/datasets/fss/FS2K_data/test/anno_test.json', help='path to att')

        # To avoid cropping, the loadSize should be the same as fineSize
        parser.set_defaults(loadSize=parser.get_default('fineSize'))
        self.isTrain = False
        return parser
