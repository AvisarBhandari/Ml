from torchvision import transforms
import torch
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from class_names import class_names
from PIL import Image
def predict_image(f:str,model:torch.nn.Module, device):
  model.eval()
  weights = EfficientNet_B0_Weights.DEFAULT
  transform = weights.transforms()
  result ={"name":[],"pred":[]}
  img = Image.open(f).convert("RGB")
  img_tensor = transform(img)

  img_tensor = img_tensor.unsqueeze(0).to(device)
  with torch.inference_mode():
    prob_logit = model(img_tensor)
    prob = (torch.softmax(prob_logit,dim=1)).squeeze()
    prob_label = torch.argmax(prob_logit,dim=1).item()
    prob_label = class_names[prob_label]
    top_probs, top_indices = torch.topk(prob, k=3)

  for i in range(len(top_probs)):
    name = class_names[top_indices[i].item()]
    percentage = round(top_probs[i].item()*100)
    result["name"].append(name)
    result["pred"].append(percentage)
  return result
