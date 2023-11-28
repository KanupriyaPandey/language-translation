# Function to load and prepare training data
def prepare_train_dataloader(train_pairs, batch_size):
    input_ids_main = np.zeros((len(train_pairs), MAX_LENGTH), dtype=np.int32)
    target_ids_main = np.zeros((len(train_pairs), MAX_LENGTH), dtype=np.int32)

    for idx, (input, target) in enumerate(train_pairs):
        input_ids = [input_language.word2index[word] for word in input.split(' ')]
        target_ids = [output_language.word2index[word] for word in target.split(' ')]

        input_ids.append(EOS_token)
        target_ids.append(EOS_token)

        input_ids_main[idx, :len(input_ids)] = input_ids
        target_ids_main[idx, :len(target_ids)] = target_ids

    input_ids_main = torch.LongTensor(input_ids_main).to(device)
    target_ids_main = torch.LongTensor(target_ids_main).to(device)

    train_data = TensorDataset(input_ids_main, target_ids_main)
    train_sampler = RandomSampler(train_data)

    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    return input_language, output_language, train_dataloader

# Function to define optimizer
def define_optimizer(encoder, decoder, learning_rate):
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)

    return encoder_optimizer, decoder_optimizer

# Function to define loss function
def define_loss():
    criterion = nn.NLLLoss()

    return criterion

# Function to train the model
def train(train_dataloader, encoder, decoder,
          encoder_optimizer, decoder_optimizer, criterion,
          epochs, print_every=10):

    loss_total = 0

    for epoch in range(1, epochs + 1):
        loss_val = 0
        for data in train_dataloader:
            input_tensor, target_tensor = data

            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            encoder_outputs, encoder_hidden = encoder(input_tensor)
            decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)

            loss = criterion(
                decoder_outputs.view(-1, decoder_outputs.size(-1)),
                target_tensor.view(-1)
            )
            loss.backward()

            encoder_optimizer.step()
            decoder_optimizer.step()

            loss_val += loss.item()

        loss_val =  loss_val/len(train_dataloader)
        loss_total += loss

        if epoch % print_every == 0:
            print_loss_avg = loss_total/print_every
            print_loss_total = 0
            print(epoch, print_loss_avg)