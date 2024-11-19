def train(model, dataloader, criterion, optimizer, num_epochs, verbose=True):
    # Training Loop
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        epoch_loss = 0

        for batch in dataloader:
            inputs = batch[0]

            # Forward pass
            outputs = model(inputs)
            
            # Compute loss
            loss = criterion(outputs, inputs)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            
            
            epoch_loss += loss.item()

        # Average loss over the epoch
        avg_loss = epoch_loss / len(dataloader)
        if verbose:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.8f}")

    return model