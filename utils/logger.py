from torch.utils.tensorboard import SummaryWriter

class TensorboardWriter:
    def __init__(self, log_dir="runs", comment=""):
        # Initialize the SummaryWriter with a log directory and optional comment
        self.writer = SummaryWriter(log_dir=log_dir, comment=comment)
    
    def log_scalar(self, tag, value, step):
        # Log a scalar value (e.g., loss, reward)
        self.writer.add_scalar(tag, value, step)
    
    def log_histogram(self, tag, values, step):
        # Log a histogram (e.g., weight distributions)
        self.writer.add_histogram(tag, values, step)
    
    def log_image(self, tag, img_tensor, step):
        # Log an image (e.g., state observations)
        self.writer.add_image(tag, img_tensor, step)
    
    def log_text(self, tag, text_string, step):
        # Log text information
        self.writer.add_text(tag, text_string, step)
    
    def close(self):
        # Close the writer
        self.writer.close()


'''
# Logging scalar values (e.g., loss)
writer.log_scalar("loss", loss_value, step)

# Logging histograms (e.g., model parameters)
writer.log_histogram("weights/layer1", model.layer1.weight, step)

# Logging images (e.g., input observations)
writer.log_image("input_state", state_image, step)

# Logging text (e.g., model summary or comments)
writer.log_text("model_info", "This is a test model", step)
'''