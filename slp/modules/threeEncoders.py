import torch.nn as nn
import torch

from slp.modules.hier_att_net import HierAttNet
from slp.util import to_device



class ThreeEncoders(nn.Module):
    def __init__(self, encoder, device, hidden_size, num_classes, merge='cat', non_blocking: bool = True):
        super(ThreeEncoders, self).__init__()
        self.therapist_encoder = encoder
        self.patient_encoder = encoder
        self.ensemble_encoder = encoder
        self.merge = merge
        self.device = device
        self.non_blocking = non_blocking
        self.fc = nn.Linear(3 * 2 * hidden_size, num_classes)
#        self.fc = nn.Linear(3 * hidden_size, num_classes)	# for non-bi


    def _merge(self, input1, input2, input3):
        if self.merge == 'cat':
            d1 = torch.cat((input1, input2), dim = 1)
            d2 = torch.cat((d1, input3), dim = 1)
            return d2

    def parse_batch(self, batch):
        therapist = to_device(batch[0],
                           device=self.device,
                           non_blocking=self.non_blocking)
        patient = to_device(batch[1],
                            device=self.device,
                            non_blocking=self.non_blocking)
        ensemble = to_device(batch[2],
                            device=self.device,
                            non_blocking=self.non_blocking)
        len_therapist = to_device(batch[3],
                            device=self.device,
                            non_blocking=self.non_blocking)
        len_patient = to_device(batch[4],
                            device=self.device,
                            non_blocking=self.non_blocking)
        len_ensemble = to_device(batch[5],
                            device=self.device,
                            non_blocking=self.non_blocking)
        targets = to_device(batch[6],
                            device=self.device,
                            non_blocking=self.non_blocking)

        return therapist, patient, ensemble, len_therapist, len_patient, len_ensemble, targets

    def forward(self, batch):
#        import pdb; pdb.set_trace()
        therapist, patient, ensemble, len_therapist, len_patient, len_ensemble, targets = self.parse_batch(batch)
        output_therapist = self.therapist_encoder(therapist, len_therapist)
        output_patient = self.patient_encoder(patient, len_patient)
        output_ensemble = self.ensemble_encoder(ensemble, len_ensemble)

        merged = self._merge(output_therapist, output_patient, output_ensemble)

        output = self.fc(merged).squeeze()

        return output, targets

