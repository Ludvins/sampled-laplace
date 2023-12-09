import torch


class _ECELoss(torch.nn.Module):
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        
        self.confidences = []
        self.predictions = []
        self.labels = []
        
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

        bin_boundaries_plot = torch.linspace(0, 1, 11)
        self.bin_lowers_plot = bin_boundaries_plot[:-1]
        self.bin_uppers_plot = bin_boundaries_plot[1:]
        
    def reset(self):
        self.confidences = []
        self.predictions = []
        self.labels = []
    
    def update(self, labels, F):
        probs = F.softmax(-1)
        conf, pred = torch.max(probs, -1)
        self.confidences.append(conf)
        self.predictions.append(pred)
        self.labels.append(labels.squeeze(-1))

    def compute(self):
        
        self.predictions = torch.cat(self.predictions, -1)
        self.labels = torch.cat(self.labels, -1)
        self.confidences = torch.cat(self.confidences, -1)
        
        
        accuracies = self.predictions.eq(self.labels)
        ece = torch.zeros(1, device=self.confidences.device)
        
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = self.confidences.gt(bin_lower.item()) * self.confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = self.confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece