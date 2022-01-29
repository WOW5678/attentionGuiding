
import torch

class MedNLIClassifier(torch.nn.Module):
    def __init__(self, nn_tokenizer, nn_bert_model, args):
        super(MedNLIClassifier, self).__init__()
        self.args = args
        self.nn_bert_model = nn_bert_model
        self.nn_tokenizer = nn_tokenizer
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(768, len(args.label2id))

    def forward(self, token_ids, mask_ids, seg_ids):
        if self.args.model_name == 'roberta-base':
            outputs = self.nn_bert_model(token_ids, attention_mask=mask_ids)
        else:
            outputs = self.nn_bert_model(token_ids, attention_mask=mask_ids, token_type_ids=seg_ids)

        output_pooled = outputs['pooler_output']
        output_pooled = self.dropout(output_pooled)
        prediction = self.classifier(output_pooled)

        if self.args.loss_type == 'task':
            return prediction
        else:
            attentions = outputs['attentions']
            attentions = torch.stack(attentions, dim=0) #[12, 64, 12, 256,256]
            attentions = torch.mean(attentions, -1)  # 对所有token求平均 [12，64,12,256]
            attentions = attentions.transpose(1, 0)  # [64,12,12,256]
            attentions = attentions.reshape(attentions.shape[0], -1, attentions.shape[-1])  # [64,144,256]

            return prediction, attentions






