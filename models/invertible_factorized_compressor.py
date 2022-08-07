import torch
import torch.nn as nn
from modules import *
from modules.invertible_blocks import INN


class InvertibleFactorizedCompressor(nn.Module):

    def __init__(self, a, h, rank) -> None:
        super(InvertibleFactorizedCompressor, self).__init__()
        self.a = a
        self.h = h
        self.device = torch.device('cuda:{:d}'.format(rank))
        self.encoder = Analysis_net(192)
        self.decoder = Synthesis_net(192)
        self.bit_estimator = BitsEstimator(192, K=5)
        self.entropy_coder = EntropyCoder(self.bit_estimator)
        self.inn = INN()

    def forward(self, inputs):
        """
        :param inputs: mini-batch
        :return: rec_imgs: 重构图像  bits_map: 累计分布函数
        """

        after_inn = self.inn.forward(inputs)

        y = self.encoder(after_inn)
        y_hat = self.quantize(y, is_train=True)
        rec_before_inn = torch.clamp(self.decoder(y_hat), 0, 2)
        rec_after_inn = self.inn.inverse(rec_before_inn)

        # R loss
        total_bits = self.bit_estimator.total_bits(y_hat)
        img_shape = rec_after_inn.size()
        bpp = total_bits / (img_shape[0] * img_shape[2] * img_shape[3])

        # D loss
        distortion = torch.mean((inputs - rec_after_inn) ** 2)
        distortion_inn = torch.mean((after_inn - rec_before_inn) ** 2)

        # total loss
        loss = bpp + self.a.Lambda * (255 ** 2) * (distortion + distortion_inn)

        # return loss, bpp, distortion, rec_imgs
        return {'Loss':loss, 'Bpp':bpp, 'Distortion':distortion, 'Distortion_inn':distortion_inn, 'Rec_images':rec_after_inn}

    def quantize(self, y, is_train=False):
        if is_train:
            uniform_noise = nn.init.uniform_(torch.zeros_like(y), -0.5, 0.5)
            if torch.cuda.is_available():
                uniform_noise = uniform_noise.to(self.device)
            y_hat = y + uniform_noise
        else:
            y_hat = torch.round(y)
        return y_hat

    def inference(self, img):
        """
        only use in test and validate
        """
        after_inn = self.inn.forward(img)
        y = self.encoder(after_inn)
        y_hat = self.quantize(y, is_train=False)
        stream, side_info = self.entropy_coder.compress(y_hat)
        y_hat_dec = self.entropy_coder.decompress(stream, side_info, y_hat.device)
        assert torch.equal(y_hat, y_hat_dec), "Entropy code decode for y_hat not consistent !"
        rec_before_inn = torch.clamp(self.decoder(y_hat), 0, 2)
        rec_after_inn = self.inn.inverse(rec_before_inn)
        bpp = len(stream) * 8 / img.shape[2] / img.shape[3]
        return rec_after_inn, bpp

if __name__ == '__main__':

    a = None
    h = None
    rank = 0
    model = InvertibleFactorizedCompressor(a, h, rank)

    for param in model.named_parameters():
        print(param[0])
