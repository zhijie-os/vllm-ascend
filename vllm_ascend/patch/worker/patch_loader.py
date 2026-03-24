import vllm
from vllm.model_executor.models.utils import AutoWeightsLoader

class AutoWeightsLoaderWithPreprocess(AutoWeightsLoader):
    def _preprocess(self):
        # when the module is quantized, transpose flag false, restore_weights
        for _, module in self.module.named_modules():
            if hasattr(module, "quant_method") and getattr(module, "_mxfp8_transformed", False):
                q_method = getattr(module.quant_method, 'quant_method', None)
                if q_method and hasattr(q_method, 'restore_weights_for_rl_loading'):
                    q_method.restore_weights_for_rl_loading(module)
        
        # when the module is not quantized, transpose flag true
        for _, v in self.module.named_parameters():
            if getattr(v, "transposed", False):
                setattr(v, "transposed", False)  # noqa: B010
                v.data = v.data.transpose(1, 2)
        
    def load_weights(self, *args, **kwargs):
        self._preprocess()
        super().load_weights(*args, **kwargs)


vllm.model_executor.models.utils.AutoWeightsLoader = AutoWeightsLoaderWithPreprocess
