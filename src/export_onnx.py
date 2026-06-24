import os 
import torch 

def export_to_onnx(model, model_name, save_dir="exports"):
    """
    Trace pytorch model and saves it as an onnx file.
    """
    # ensure the export dir exists 
    os.makedirs(save_dir, exist_ok = True)

    # put model in eval mode on cpu (ONNX export are done on CPU)
    model.eval()
    model.cpu()

    # create dummy input 
    dummy_input = torch.randn(1, 3, 224, 224)

    # define the output path 
    export_path = os.path.join(save_dir, f"{model_name}.onnx")

    print(f"Exporting {model_name} to {export_path}...")

    # export 
    torch.onnx.export(
        model, # model we want to export
        dummy_input,  # dummy data to trace 
        export_path,
        export_params = True, # save the weights inside the file
        opset_version= 18, # onnx syntax version
        do_constant_folding = True, # optimize graph during export 
        input_names = ['input'], # name the input pipe 
        output_names = ['output'], # name the output pipe 

        # tell onnx that batch size can change
        dynamic_axes = {
            'input' : {0: 'batch_size'},
            'output' : {0: 'batch_size'},
        }
    )

    print("Export complete!")
    return export_path