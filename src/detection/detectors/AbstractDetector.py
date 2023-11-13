import torch


class AbstractDetector(object):

    def __init__(self, **kvargs):
        print('Initializing Detector...')

    def predict(self, txt):
        tokens = self.tokenizer.encode(txt, truncation=True, max_length=512)
        tokens = torch.Tensor(tokens)

        tokens = tokens.unsqueeze(0).cuda().long()

        mask = torch.ones_like(tokens).cuda().long()

        logits = self.model(tokens, attention_mask=mask)

        probs = logits[0].softmax(dim=-1)

        probs = probs.detach().cpu().flatten().numpy()

        return probs
