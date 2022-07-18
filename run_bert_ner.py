from __future__ import absolute_import, division, print_function

from datetime import datetime

import wandb
from pytorch_transformers import (AdamW, WarmupLinearSchedule)
from seqeval.metrics.sequence_labeling import precision_recall_fscore_support, get_entities
from torch import nn
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from tqdm import tqdm, trange
from seqeval.metrics import classification_report
from tools.deepke.name_entity_re.standard import *


wandb.init(project="DoctorKG-NER-Example")


# 自定义 NER 任务
class TrainNer(BertForTokenClassification):

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, valid_ids=None,
                attention_mask_label=None, device=None):
        sequence_output = self.bert(input_ids, token_type_ids, attention_mask, head_mask=None)[0]
        batch_size, max_len, feat_dim = sequence_output.shape
        valid_output = torch.zeros(batch_size, max_len, feat_dim, dtype=torch.float32, device=device)
        for i in range(batch_size):
            jj = -1
            for j in range(max_len):
                if valid_ids[i][j].item() == 1:
                    jj += 1
                    valid_output[i][jj] = sequence_output[i][j]
        sequence_output = self.dropout(valid_output)
        logits = self.classifier(sequence_output)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=0)
            if attention_mask_label is not None:
                active_loss = attention_mask_label.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits


def main():
    # 训练设置 #
    device = torch.device("cpu")
    gradient_accumulation_steps = 1
    output_dir = "checkpoints"
    adam_epsilon = 1e-8
    learning_rate = 5e-5
    bert_model = "bert-base-chinese"
    data_dir = "data"  # 其下的数据务必命名为 train.txt/test.txt/valid.txt
    eval_batch_size = 32
    eval_on = "test"
    task_name = "ner"
    seed = 42
    do_lower_case = True
    warmup_proportion = 0.1
    weight_decay = 0.01
    max_grad_norm = 1.0
    max_seq_length = 128
    num_train_epochs = 30  # the number of training epochs
    train_batch_size = 32
    every_n_step = 1  # 每 n 步进行一次 eval 测试
    # 参数检查 #
    if gradient_accumulation_steps < 1:
        raise ValueError(
            "Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(gradient_accumulation_steps))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # 训练与测试前的准备
    processor = NerProcessor()
    label_list = processor.get_labels()
    train_batch_size = train_batch_size // gradient_accumulation_steps
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    num_labels = len(label_list) + 1
    # 默认不会载入之前的模型进行训练
    # model = TrainNer.from_pretrained(output_dir)
    # tokenizer = BertTokenizer.from_pretrained(output_dir, do_lower_case=do_lower_case)
    tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=do_lower_case)
    config = BertConfig.from_pretrained(bert_model, num_labels=num_labels, finetuning_task=task_name)
    model = TrainNer.from_pretrained(bert_model, from_tf=False, config=config)
    model.to(device)
    # CSV 文件记录参数
    global_step = 0
    csv_eval_result_file = open(os.path.join(output_dir, "eval_results.csv"), "w")
    csv_eval_result_file_fieldnames = ['date', 'step', 'eval_precision', 'eval_recall', 'eval_f1', 'train_loss']
    csv_eval_result_file_writer = csv.DictWriter(csv_eval_result_file, fieldnames=csv_eval_result_file_fieldnames)
    csv_eval_result_file_writer.writeheader()
    csv_eval_result_file_to_write_dict = {
        "date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "step": global_step,
        "eval_precision": None,
        "eval_recall": None,
        "eval_f1": None,
        "train_loss": None,
    }
    for _ in trange(int(num_train_epochs), desc="Epoch"):
        # Training!
        model.train()
        train_examples = processor.get_train_examples(data_dir)
        num_train_optimization_steps = int(
            len(train_examples) / train_batch_size / gradient_accumulation_steps) * num_train_epochs
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        warmup_steps = int(warmup_proportion * num_train_optimization_steps)
        optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps, t_total=num_train_optimization_steps)
        train_features = convert_examples_to_features(train_examples, label_list, max_seq_length, tokenizer)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        all_valid_ids = torch.tensor([f.valid_ids for f in train_features], dtype=torch.long)
        all_lmask_ids = torch.tensor([f.label_mask for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_valid_ids,
                                   all_lmask_ids)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=train_batch_size)
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            model.train()
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids, valid_ids, l_mask = batch
            loss = model(input_ids, segment_ids, input_mask, label_ids, valid_ids, l_mask, device)
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            tr_loss += loss.item()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1
            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
            global_step += 1
            # print("train_loss: " + str(tr_loss / nb_tr_steps))
            # Evaling!
            if global_step % every_n_step == 0:
                # 测试
                model.eval()
                if eval_on == "dev":
                    eval_examples = processor.get_dev_examples(data_dir)
                elif eval_on == "test":
                    eval_examples = processor.get_test_examples(data_dir)
                else:
                    raise ValueError("eval on dev or test set only")
                eval_features = convert_examples_to_features(eval_examples, label_list, max_seq_length, tokenizer)
                eval_all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
                eval_all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
                eval_all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
                eval_all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
                eval_all_valid_ids = torch.tensor([f.valid_ids for f in eval_features], dtype=torch.long)
                eval_all_lmask_ids = torch.tensor([f.label_mask for f in eval_features], dtype=torch.long)
                eval_data = TensorDataset(eval_all_input_ids, eval_all_input_mask, eval_all_segment_ids,
                                          eval_all_label_ids, eval_all_valid_ids, eval_all_lmask_ids)
                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=eval_batch_size)
                y_true = []
                y_pred = []
                label_map = {i: label for i, label in enumerate(label_list, 1)}
                # print(label_map)
                for input_ids, input_mask, segment_ids, label_ids, valid_ids, l_mask in tqdm(eval_dataloader, desc="Evaluating"):
                    input_ids = input_ids.to(device)
                    input_mask = input_mask.to(device)
                    segment_ids = segment_ids.to(device)
                    valid_ids = valid_ids.to(device)
                    label_ids = label_ids.to(device)
                    l_mask = l_mask.to(device)
                    with torch.no_grad():
                        logits = model(input_ids, segment_ids, input_mask, valid_ids=valid_ids,
                                       attention_mask_label=l_mask, device=device)
                    logits = torch.argmax(F.log_softmax(logits, dim=2), dim=2)
                    logits = logits.detach().cpu().numpy()
                    label_ids = label_ids.to('cpu').numpy()
                    for i, label in enumerate(label_ids):
                        temp_1 = []
                        temp_2 = []
                        for j, m in enumerate(label):
                            if j == 0:
                                continue
                            elif label_ids[i][j] == len(label_map):  # attention: [SEP]是最后一个（编号为len(label_map)）表示分隔
                                y_true.append(temp_1)
                                y_pred.append(temp_2)
                                break
                            else:
                                temp_1.append(label_map[label_ids[i][j]])
                                if logits[i][j] != 0:
                                    temp_2.append(label_map[logits[i][j]])
                                else:
                                    temp_2.append('0')
                # print(len(y_true))
                # print(len(y_pred))
                report = classification_report(y_true, y_pred, digits=4)
                # 处理结果
                p, r, f1, s = precision_recall_fscore_support(y_true, y_pred)
                target_names_true = {type_name for type_name, _, _ in get_entities(y_true)}
                target_names_pred = {type_name for type_name, _, _ in get_entities(y_pred)}
                target_names = sorted(target_names_true | target_names_pred)
                for row in zip(target_names, p, r, f1, s):
                    # print(row)
                    if row[0] == 'SCH':  # 对自己标签的结果，记载一下
                        csv_eval_result_file_to_write_dict['train_loss'] = tr_loss / nb_tr_steps
                        csv_eval_result_file_to_write_dict['eval_precision'] = row[1]
                        csv_eval_result_file_to_write_dict['eval_recall'] = row[2]
                        csv_eval_result_file_to_write_dict['eval_f1'] = row[3]
                        csv_eval_result_file_to_write_dict['date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        csv_eval_result_file_to_write_dict['step'] = global_step
                        csv_eval_result_file_writer.writerow(csv_eval_result_file_to_write_dict)
                        csv_eval_result_file.flush()
                        # print(str(csv_eval_result_file_to_write_dict))
                        # wandb 每 call 一次会增加一次 step 哦，所以只在这里调用
                        wandb.log({
                            "acc": row[1],
                            "recall": row[2],
                            "f1": row[3],
                            "train_loss": tr_loss / nb_tr_steps
                        })
                output_eval_file = os.path.join(output_dir, "eval_results.txt")
                with open(output_eval_file, "w") as writer:
                    writer.write(report)
    # Save a trained model and the associated configuration
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    csv_eval_result_file.close()


if __name__ == '__main__':
    main()
