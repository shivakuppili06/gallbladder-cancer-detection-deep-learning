import torch
import torch.nn.functional as F
import numpy as np
import cv2
import base64


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        # Disable inplace activations to prevent autograd inplace view error (e.g. DenseNet121)
        for module in self.model.modules():
            if hasattr(module, 'inplace'):
                module.inplace = False

        self.gradients = None
        self.activations = None
        self._fwd_handle = target_layer.register_forward_hook(self._save_activation)
        self._bwd_handle = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.clone().detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].clone().detach()

    def generate(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        # Perform backward pass (no retain_graph to immediately release computation graph memory)
        output[0, class_idx].backward()

        if self.gradients is None or self.activations is None:
            raise RuntimeError("GradCAM failed to capture activations or gradients.")

        gradients = self.gradients[0]        # [C, H, W]
        activations = self.activations[0]    # [C, H, W]
        weights = gradients.mean(dim=(1, 2)) # [C]

        # Vectorized calculation (instant C++ matrix multiplication, zero memory overhead)
        cam = torch.sum(weights[:, None, None] * activations, dim=0)

        cam = F.relu(cam).cpu().numpy()
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        cam = cv2.resize(cam, (224, 224))
        return cam, class_idx

    def remove_hooks(self):
        try:
            self._fwd_handle.remove()
        except Exception:
            pass
        try:
            self._bwd_handle.remove()
        except Exception:
            pass


def overlay_heatmap(cam, original_bgr_224):
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original_bgr_224, 0.55, heatmap, 0.45, 0)
    _, buf = cv2.imencode(".jpg", overlay)
    return base64.b64encode(buf).decode("utf-8")


def get_target_layer(model, arch_name):
    """Confirmed against train.py's build_model() for all architectures."""
    name = arch_name.lower()
    if "efficientnet" in name:
        return model.features[-1]
    if "resnet" in name:
        return model.layer4[-1]
    if "mobilenet" in name:
        return model.features[-1]
    if "densenet" in name:
        return model.features.denseblock4
    raise ValueError(f"No Grad-CAM target layer mapped for {arch_name}")
