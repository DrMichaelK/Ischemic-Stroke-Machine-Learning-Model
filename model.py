# model.py

import torch
import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM

class ImageToTextProjector(nn.Module):
    def __init__(self, image_embedding_dim, text_embedding_dim):
        super(ImageToTextProjector, self).__init__()
        self.fc = nn.Linear(image_embedding_dim, text_embedding_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.fc(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x

class CombinedModel(nn.Module):
    def __init__(self, video_model, report_generator, num_classes, projector, tokenizer):
        super(CombinedModel, self).__init__()
        self.video_model = video_model
        self.report_generator = report_generator
        self.classifier = nn.Linear(512, num_classes)
        self.projector = projector
        self.dropout = nn.Dropout(p=0.5)
        self.tokenizer = tokenizer  # Store tokenizer

    def forward(self, images, labels=None):
        video_embeddings = self.video_model(images)
        video_embeddings = self.dropout(video_embeddings)
        class_outputs = self.classifier(video_embeddings)
        projected_embeddings = self.projector(video_embeddings)
        encoder_inputs = projected_embeddings.unsqueeze(1)

        if labels is not None:
            outputs = self.report_generator(
                inputs_embeds=encoder_inputs,
                labels=labels
            )
            gen_loss = outputs.loss
            generated_report = None
        else:
            generated_report_ids = self.report_generator.generate(
                inputs_embeds=encoder_inputs,
                max_length=512,
                num_beams=4,
                early_stopping=True
            )
            generated_report = self.tokenizer.batch_decode(
                generated_report_ids, skip_special_tokens=True
            )
            gen_loss = None

        return class_outputs, generated_report, gen_loss
