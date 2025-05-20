import torch
import torch.nn as nn

class CustomCoxLoss(nn.Module):
    def forward(self, risk_scores, times, events):
        if events.sum() == 0:
            return torch.tensor(0.0, requires_grad=True, device=risk_scores.device)
        
        # Sort by descending time
        order = torch.argsort(times, descending=True)
        times = times[order]
        events = events[order]
        risk_scores = risk_scores[order]

        log_cumsum = torch.logcumsumexp(risk_scores, dim=0)
        loss = -(risk_scores - log_cumsum) * events
        return loss.sum() / events.sum()
    

#Custom C-Index
def concordance_index(risk_scores, times, events):
    n = 0
    n_correct = 0
    risk_scores = -risk_scores.detach().cpu().numpy()
    times = times.detach().cpu().numpy()
    events = events.detach().cpu().numpy()

    for i in range(len(times)):
        for j in range(len(times)):
            if times[i] < times[j] and events[i] == 1:
                n += 1
                if risk_scores[i] > risk_scores[j]:
                    n_correct += 1
                elif risk_scores[i] == risk_scores[j]:
                    n_correct += 0.5
    return n_correct / n if n > 0 else 0
