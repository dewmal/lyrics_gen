# import data loader
import pandas as pd
import numpy as np

# Data pre processing
import spacy

# Computation
import torch.nn as nn
import torch
import torch.optim as optim
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Events, Engine
from ignite.handlers import ModelCheckpoint, Timer
from ignite.metrics import RunningAverage

source_limit = 100

model_name = "gan_model_3"
model_version = "v1"

learning_rate = 0.001
beta_1 = 0.99
batch_size = 1
z_dim = 1
feature_size = 300

#
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'
print(device)
from gen import Generator, Descriminator

nlp = spacy.load("en_core_web_md")

lyrics = pd.read_csv('data/380000-lyrics-from-metrolyrics.zip')
lyrics.dropna(inplace=True)

from torch.utils.data import Dataset, DataLoader


# nlp.add_pipe(nlp.create_pipe('sentencizer'))
class LyricsDataSet(Dataset):

    def __init__(self, df, train=True):
        self.train = train
        self.df = df

    def __getitem__(self, idx):
        data_y = self.df.iloc[idx, 5:]
        data_x = self.df.iloc[idx, :5]
        song = data_y.values[0]
        # song = nlp(song)
        song_vec_sents = []
        try:
            for sent in song.split("\n"):
                sent = nlp(sent)
                song_vec_sents.append(sent.vector)
        except:
            print(song)
        song_vec_sents = np.array(song_vec_sents)

        return song_vec_sents

    def __len__(self):
        return source_limit  # len(self.df)


dt_x = LyricsDataSet(lyrics).__getitem__(50)
print(dt_x.shape)

netG = Generator(1, feature_size).to(device)
netD = Descriminator(1, feature_size).to(device)

print(netG)
print(netD)

# criterion
bce = nn.BCELoss()

# optimizers
optimizerG = optim.Adam(netG.parameters(), lr=learning_rate, betas=(beta_1, 0.999))
optimizerD = optim.Adam(netD.parameters(), lr=learning_rate, betas=(beta_1, 0.999))

# misc
real_labels = torch.ones(batch_size, device=device)
fake_labels = torch.zeros(batch_size, device=device)
fixed_noise = torch.randn(batch_size, z_dim, 1, 1, device=device)


def get_noise(batch_size, z_dim):
    return torch.randn((1, batch_size, feature_size), device=device)


# misc
real_labels = torch.ones((1, 2), device=device)
fake_labels = torch.zeros((1, 2), device=device)
fixed_noise = torch.randn(1, z_dim, 1, 1, device=device)  # fake_label = torch.from_numpy([[1,0]])


def step(engine, batch):
    real = batch
    real = real.to(device)
    # real = torch.reshape(real,(real.shape[1],real.shape[2]))
    # print(real.shape)
    # for real_row in real:
    #  print(real_row.shape)

    netD.zero_grad()

    # train with real
    # print(real.shape)
    output = netD(real)
    # print("---")
    # print(output.shape, real_labels.shape)

    errD_real = bce(output, real_labels)
    D_x = output.mean().item()

    errD_real.backward()

    # get fake image from generator
    noise = get_noise(real.shape[1], 1)
    fake = netG(noise)

    # print(fake.shape)

    # train with fake
    output = netD(fake.detach())
    errD_fake = bce(output, fake_labels)
    D_G_z1 = output.mean().item()

    errD_fake.backward()

    # gradient update
    errD = errD_real + errD_fake
    optimizerD.step()

    # -----------------------------------------------------------
    # (2) Update G network: maximize log(D(G(z)))
    netG.zero_grad()

    # Update generator. We want to make a step that will make it more likely that discriminator outputs "real"
    output = netD(fake)
    errG = bce(output, real_labels)
    D_G_z2 = output.mean().item()

    errG.backward()

    # gradient update
    optimizerG.step()

    return {
        'errD': errD.item(),
        'errG': errG.item(),
        'D_x': D_x,
        'D_G_z1': D_G_z1,
        'D_G_z2': D_G_z2
    }


trainer = Engine(step)
checkpoint_handler = ModelCheckpoint(f"models/{model_name}", model_version, save_interval=1, n_saved=10,
                                     save_as_state_dict=True,
                                     require_empty=False)
timer = Timer(average=True)

# attach running average metrics
monitoring_metrics = ['errD', 'errG', 'D_x', 'D_G_z1', 'D_G_z2']
alpha = 0.99
RunningAverage(alpha=alpha, output_transform=lambda x: x['errD']).attach(trainer, 'errD')
RunningAverage(alpha=alpha, output_transform=lambda x: x['errG']).attach(trainer, 'errG')
RunningAverage(alpha=alpha, output_transform=lambda x: x['D_x']).attach(trainer, 'D_x')
RunningAverage(alpha=alpha, output_transform=lambda x: x['D_G_z1']).attach(trainer, 'D_G_z1')
RunningAverage(alpha=alpha, output_transform=lambda x: x['D_G_z2']).attach(trainer, 'D_G_z2')

# attach progress bar
pbar = ProgressBar()
pbar.attach(trainer, metric_names=monitoring_metrics)


# Print Time
@trainer.on(Events.EPOCH_COMPLETED)
def print_times(engine):
    pbar.log_message('Epoch {} done. Time per batch: {:.3f}[s]'.format(engine.state.epoch, timer.value()))
    timer.reset()


trainer.add_event_handler(event_name=Events.EPOCH_COMPLETED, handler=checkpoint_handler,
                          to_save={
                              'netG': netG,
                              'netD': netD
                          })

if __name__ == '__main__':
    x_y_data_set = LyricsDataSet(lyrics)
    data_loader = DataLoader(x_y_data_set, batch_size=1,
                             shuffle=True)
    trainer.run(data_loader, 100)
