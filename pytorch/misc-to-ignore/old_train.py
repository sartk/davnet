


if configs['print_progress']:
logging.info('\nEpoch {}/{}:\n'.format(epoch+1, configs['num_epochs']))
train_bal = tqdm(train_bal)
train_unbal = tqdm(train_unbal)


for group in ['train', 'val']:

for group in []



for i, (img, seg_label, domain_label) in enumerate(train_dl):



if configs['half_precision']:
img = img.type(torch.HalfTensor).cuda(non_blocking=True)
else:
img = img.type(torch.FloatTensor).cuda(non_blocking=True)

optimizer.zero_grad()

if group == 'bal':
seg_pred, domain_pred = model(img, lamb, False)
seg_loss = F_seg_loss(seg_pred, seg_label)
domain_loss = F_domain_loss(domain_pred, domain_label)
err = seg_loss + domain_loss

elif group == 'unbal':
if torch.argmax(domain_label).items() == 0:
seg_pred = model(img, lamb, True)
seg_loss = F_seg_loss(seg_pred, seg_label)
domain_loss = F_domain_loss(domain_pred, domain_label)
err = seg_loss + domain_loss

if phase == 'train':
err.backward()
optimizer.step()



sample_count += img.size(0)
running_loss_domain += domain_loss.item() * img.size(0)  # smaller batches count less
running_acc_domain += (domain_pred.argmax(-1) == domain_label).sum().item()  # num corrects
running_loss_seg += seg_loss.item() * img.size(0)

if configs['print_progress']:
# inspect gradient L2 norm
total_norm = torch.zeros(1).cuda()
for name, param in model.named_parameters():
    try:
        total_norm += param.grad.data.norm(2)**2
    except:
        pass
total_norm = total_norm**(1/2)
writer.add_scalar('grad_L2_norm', total_norm, configs['step'])

epoch_train_loss = running_loss / sample_count
epoch_train_acc = running_acc / sample_count

# reduce lr
if configs['decay_steps'] > 0 or configs['decay_milestones'][0] > 0:
lr_decay.step()
else:  # reduce on plateau, evaluate to keep track of acc in each process
epoch_valid_loss, epoch_valid_acc = evaluate(model, dataloaders['valid'], args)
lr_decay.step(epoch_valid_acc[0])

if configs['print_progress']:  # only validate using process 0
if epoch_valid_loss is None:  # check if process 0 already validated
epoch_valid_loss, epoch_valid_acc = evaluate(model, dataloaders['valid'], args)

logging.info('\n[Train] loss: {:.4f} - acc: {:.4f} | [Valid] loss: {:.4f} - acc: {:.4f} - acc_topk: {:.4f}'.format(
epoch_train_loss, epoch_train_acc,
epoch_valid_loss, epoch_valid_acc[0], epoch_valid_acc[1]))

epoch_valid_acc = epoch_valid_acc[0]  # discard top k acc
writer.add_scalars('epoch_loss', {'train': epoch_train_loss,
                                   'valid': epoch_valid_loss}, epoch+1)
writer.add_scalars('epoch_acc', {'train': epoch_train_acc,
                                  'valid': epoch_valid_acc}, epoch+1)
writer.add_scalars('epoch_error', {'train': 1-epoch_train_acc,
                                    'valid': 1-epoch_valid_acc}, epoch+1)

if epoch_valid_acc >= best_valid_acc:
patience_counter = 0
best_epoch = epoch + 1
best_valid_acc = epoch_valid_acc
best_valid_loss = epoch_valid_loss
# saving using process (rank) 0 only as all processes are in sync
torch.save(model.state_dict(), configs['checkpoint_dir'])
else:
patience_counter += 1
if patience_counter == (configs['patience']-10):
    logging.info('\nPatience counter {}/{}.'.format(
        patience_counter, configs['patience']))
elif patience_counter == configs['patience']:
    logging.info('\nEarly stopping... no improvement after {} Epochs.'.format(
        configs['patience']))
    break
epoch_valid_loss = None  # reset loss

gc.collect()  # release unreferenced memory

if configs['print_progress']:
time_elapsed = time.time() - since
logging.info('\nTraining time: {:.0f}m {:.0f}s'.format(
time_elapsed // 60, time_elapsed % 60))

model.load_state_dict(torch.load(configs['checkpoint_dir']))  # load best model

test_loss, test_acc = evaluate(model, dataloaders['test'], args)

logging.info('\nBest [Valid] | epoch: {} - loss: {:.4f} - acc: {:.4f}'.format(
best_epoch, best_valid_loss, best_valid_acc))
logging.info('[Test] loss {:.4f} - acc: {:.4f} - acc_topk: {:.4f}'.format(
test_loss, test_acc[0], test_acc[1]))
