import torch
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import torch.nn.functional as F

class APDrawingGANModel(BaseModel):
    def name(self):
        return 'APDrawingGANModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        # changing the default values
        parser.set_defaults(pool_size=0, no_lsgan=True, norm='batch')# no_lsgan=True, use_lsgan=False
        parser.set_defaults(dataset_mode='aligned')

        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake', 'G_VGG', 'G_CLS', 'G_local']
        self.loss_names.append('D_real_local')
        self.loss_names.append('D_fake_local')
        self.loss_names.append('G_GAN_local')
        if self.isTrain and self.opt.no_l1_loss:
            self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'D']

        self.loss_names.append('G')
        print('loss_names', self.loss_names)
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        if self.opt.use_local:
            self.visual_names += ['fake_B1']
        if not self.isTrain and self.opt.save2:
            self.visual_names = ['real_A', 'fake_B']
        print('visuals', self.visual_names)
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['D', 'D_Cls']
        else:  # during test time, only load Gs
            self.model_names = ['G']
            self.auxiliary_model_names = []
        if self.opt.use_local:
            self.model_names += ['GLEyel','GLEyer','GLNose','GLMouth','GLHair','GLBG','GCombine']
        print('model_names', self.model_names)

        if self.isTrain:
            # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            use_sigmoid = opt.no_lsgan
            self.netD_Cls = networks.define_D(opt.input_nc, opt.ndf*2, 'basic_cls', 1, opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, 'multiscale',2,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)
            print('netD', opt.netD, opt.n_layers_D)
            
            if self.opt.discriminator_local:
                self.netDLEyel = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, 'multiscale',1,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)
                self.netDLEyer = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, 'multiscale',1,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)
                self.netDLNose = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, 'multiscale',1,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)
                self.netDLMouth = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, 'multiscale',1,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)
                self.netDLHair = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, 'multiscale',1,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)
                self.netDLBG = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, 'multiscale',1,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.opt.use_local:
            self.netGLEyel = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, 'global', opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netGLEyer = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, 'global', opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netGLNose = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, 'global', opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netGLMouth = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, 'global', opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netGLHair = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, 'global', opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netGLBG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, 'global', opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netGCombine = networks.define_G(opt.output_nc, opt.output_nc, opt.ngf, 'local', opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        

        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionCls = torch.nn.CrossEntropyLoss()
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionFeat = torch.nn.L1Loss()
            self.criterionL2 = torch.nn.MSELoss()
            self.TVloss = networks.TVLoss().to(self.device)
            self.criterionVGG = networks.VGGLoss(self.device)

            # initialize optimizers
            self.optimizers = []
            G_params = list(self.netGLEyel.parameters()) + list(self.netGLEyer.parameters()) + list(self.netGLNose.parameters()) + list(self.netGLMouth.parameters()) + list(self.netGLHair.parameters()) + list(self.netGLBG.parameters()) + list(self.netGCombine.parameters()) 
            print('G_params 8 components')
            self.optimizer_G = torch.optim.Adam(G_params,
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            if not self.opt.discriminator_local:
                print('D_params 1 components')
                self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt.lr * 0.01, betas=(opt.beta1, 0.999))
            else:
                D_params = list(self.netD.parameters()) + list(self.netDLEyel.parameters()) +list(self.netDLEyer.parameters()) + list(self.netDLNose.parameters()) + list(self.netDLMouth.parameters()) + list(self.netDLHair.parameters()) + list(self.netDLBG.parameters())
                print('D_params 7 components')
                self.optimizer_D = torch.optim.Adam(D_params,
                                                lr=opt.lr * 0.01, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
        
    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.real_B_style = input['style'].to(self.device)
        self.real_B_label = input['label'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        if self.opt.use_local:
            self.real_A_eyel = input['eyel_A'].to(self.device)
            self.real_A_eyer = input['eyer_A'].to(self.device)
            self.real_A_nose = input['nose_A'].to(self.device)
            self.real_A_mouth = input['mouth_A'].to(self.device)
            self.real_B_eyel = input['eyel_B'].to(self.device)
            self.real_B_eyer = input['eyer_B'].to(self.device)
            self.real_B_nose = input['nose_B'].to(self.device)
            self.real_B_mouth = input['mouth_B'].to(self.device)
            self.center = input['center']
            self.real_A_hair = input['hair_A'].to(self.device)
            self.real_B_hair = input['hair_B'].to(self.device)
            self.real_A_bg = input['bg_A'].to(self.device)
            self.real_B_bg = input['bg_B'].to(self.device)
            self.mask = input['mask'].to(self.device) # mask for non-eyes,nose,mouth
            self.mask2 = input['mask2'].to(self.device) # mask for non-bg
        

    def forward(self):
        # EYES, NOSE, MOUTH
        fake_B_eyel = self.netGLEyel(self.real_A_eyel)
        fake_B_eyer = self.netGLEyer(self.real_A_eyer)
        fake_B_nose = self.netGLNose(self.real_A_nose)
        fake_B_mouth = self.netGLMouth(self.real_A_mouth)
        self.fake_B_nose = fake_B_nose
        self.fake_B_eyel = fake_B_eyel
        self.fake_B_eyer = fake_B_eyer
        self.fake_B_mouth = fake_B_mouth
            
        # HAIR, BG AND PARTCOMBINE
        fake_B_hair = self.netGLHair(self.real_A_hair)
        fake_B_bg = self.netGLBG(self.real_A_bg)
        self.fake_B_hair = self.masked(fake_B_hair,self.mask*self.mask2)
        self.fake_B_bg = self.masked(fake_B_bg,self.inverse_mask(self.mask2))
        self.fake_B1 = self.partCombiner2_bg(fake_B_eyel,fake_B_eyer,fake_B_nose,fake_B_mouth,fake_B_hair,fake_B_bg,self.mask*self.mask2,self.inverse_mask(self.mask2),self.opt.comb_op)
            
        # FUSION NET
        self.fake_B = self.netGCombine(self.fake_B1, self.real_B_style)

        
    def backward_D(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        fake_AB = self.fake_AB_pool.query(torch.cat((self.real_A, self.fake_B), 1)) # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD.forward(fake_AB.detach())
        _, pred_real_cls = self.netD_Cls(self.real_B)
        loss_D_real_cls = self.criterionCls(pred_real_cls, self.real_B_label)
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        if self.opt.discriminator_local:
            fake_AB_parts = self.getLocalParts(fake_AB)
            local_names = ['DLEyel','DLEyer','DLNose','DLMouth','DLHair','DLBG']
            self.loss_D_fake_local = 0
            for i in range(len(fake_AB_parts)):
                net = getattr(self, 'net' + local_names[i])
                pred_fake_tmp = net.forward(fake_AB_parts[i].detach())
                addw = self.getaddw(local_names[i])
                self.loss_D_fake_local = self.loss_D_fake_local + self.criterionGAN(pred_fake_tmp, False) * addw
            self.loss_D_fake = self.loss_D_fake + self.loss_D_fake_local

        # Real
        _, pred_fake_cls = self.netD_Cls(self.fake_B.detach())
        loss_D_fake_cls = self.criterionCls(pred_fake_cls, self.real_B_label)
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD.forward(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        if self.opt.discriminator_local:
            real_AB_parts = self.getLocalParts(real_AB)
            local_names = ['DLEyel','DLEyer','DLNose','DLMouth','DLHair','DLBG']
            self.loss_D_real_local = 0
            for i in range(len(real_AB_parts)):
                net = getattr(self, 'net' + local_names[i])
                pred_real_tmp = net.forward(real_AB_parts[i])
                addw = self.getaddw(local_names[i])
                self.loss_D_real_local = self.loss_D_real_local + self.criterionGAN(pred_real_tmp, True) * addw
            self.loss_D_real = self.loss_D_real + self.loss_D_real_local
        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5 + (loss_D_real_cls + loss_D_fake_cls) * 0.5

        self.loss_D.backward()

    def backward_G(self):
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD.forward(fake_AB)
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD.forward(real_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        self.loss_G_GAN_local = 0
        self.loss_G_local = 0
        ###local feat loss
        if self.opt.discriminator_local:
            fake_AB_parts = self.getLocalParts(fake_AB)
            real_AB_parts = self.getLocalParts(real_AB)
            local_names = ['DLEyel','DLEyer','DLNose','DLMouth','DLHair','DLBG']
            for i in range(len(fake_AB_parts)):
                net = getattr(self, 'net' + local_names[i])
                pred_fake_tmp = net.forward(fake_AB_parts[i])
                pred_real_tmp = net.forward(real_AB_parts[i])
                addw = self.getaddw(local_names[i])
                self.loss_G_GAN_local = self.loss_G_GAN_local + self.criterionGAN(pred_fake_tmp, True) * addw
                if self.opt.use_local and not self.opt.no_G_local_loss:
                    feat_weights=4.0/4.0
                    D_weights = 1.0 / 1.0
                    for k in range(1):
                        for j in range(len(pred_fake_tmp[k])-1):
                            self.loss_G_local += D_weights * feat_weights * self.criterionL1(pred_fake_tmp[k][j], pred_real_tmp[k][j].detach()) * self.opt.lambda_local * addw
        
        ###local l1 vgg loss###
        if self.opt.use_local and not self.opt.no_G_local_loss:
            local_names = ['eyel','eyer','nose','mouth','hair','bg']
            for i in range(len(local_names)):
                fakeblocal = getattr(self, 'fake_B_' + local_names[i])
                realblocal = getattr(self, 'real_B_' + local_names[i])
                addw = self.getaddw(local_names[i])
                self.loss_G_local += self.criterionVGG(fakeblocal,realblocal)* self.opt.lambda_local * addw * 0.5
                self.loss_G_local += self.criterionL1(fakeblocal,realblocal)* self.opt.lambda_local * addw
        
        ###global l1 vgg feat loss####
        if not self.opt.no_l1_loss:
            self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
            self.loss_G_VGG = (self.criterionVGG(self.fake_B, self.real_B) + self.criterionVGG(F.interpolate(self.fake_B, scale_factor=0.25, recompute_scale_factor=True), F.interpolate(self.real_B, scale_factor=0.25, recompute_scale_factor=True)) + self.criterionVGG(F.interpolate(self.fake_B, scale_factor=0.5, recompute_scale_factor=True), F.interpolate(self.real_B, scale_factor=0.5, recompute_scale_factor=True)))  *self.opt.lambda_L1 * 0.5/ 3.0
            feat_weights=4.0/4.0
            D_weights = 1.0 / 2.0
            #self.loss_G_L1 = 0
            for i in range(2):
                for j in range(len(pred_fake[i])-1):
                    self.loss_G_L1 += 1.0 * D_weights * feat_weights * self.criterionL1(pred_fake[i][j], pred_real[i][j].detach()) * self.opt.lambda_L1
            
        _, pred_fake_cls = self.netD_Cls(self.fake_B)
        self.loss_G_CLS = self.criterionCls(pred_fake_cls, self.real_B_label)      

        self.loss_G = self.loss_G_GAN + self.loss_G_GAN_local
        if 'G_L1' in self.loss_names:
            self.loss_G = self.loss_G + self.loss_G_L1
        if 'G_VGG' in self.loss_names:
            self.loss_G = self.loss_G + self.loss_G_VGG
        if 'G_CLS' in self.loss_names:
            self.loss_G = self.loss_G + self.loss_G_CLS
        if 'G_local' in self.loss_names:
            self.loss_G = self.loss_G + self.loss_G_local


        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        # update D
        self.set_requires_grad(self.netD, True) # enable backprop for D
        if self.opt.discriminator_local:
            self.set_requires_grad(self.netDLEyel, True)
            self.set_requires_grad(self.netDLEyer, True)
            self.set_requires_grad(self.netDLNose, True)
            self.set_requires_grad(self.netDLMouth, True)
            self.set_requires_grad(self.netDLHair, True)
            self.set_requires_grad(self.netDLBG, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        # update G
        self.set_requires_grad(self.netD, False) # D requires no gradients when optimizing G
        if self.opt.discriminator_local:
            self.set_requires_grad(self.netDLEyel, False)
            self.set_requires_grad(self.netDLEyer, False)
            self.set_requires_grad(self.netDLNose, False)
            self.set_requires_grad(self.netDLMouth, False)
            self.set_requires_grad(self.netDLHair, False)
            self.set_requires_grad(self.netDLBG, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
