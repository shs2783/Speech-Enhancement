import os
import logging
import torchaudio

def get_logger(name,
               format_str="%(asctime)s [%(pathname)s:%(lineno)s - %(levelname)s ] %(message)s",
               date_format='%Y-%m-%d %H:%M:%S',
               file=False):

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    handler = logging.FileHandler(name) if file else logging.StreamHandler()
    handler.setLevel(logging.INFO)

    formatter = logging.Formatter(fmt=format_str, datefmt=date_format)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger

def show_params(model):
    print("=" * 40, "Model Parameters", "=" * 40)
    num_params = 0
    for module_name, m in model.named_modules():
        if module_name == '':
            for name, params in m.named_parameters():
                print(name, params.size(), end=' ')

                i = params.numel()
                if 'weight' in name:
                    print('>'.rjust(20, '-'), i)
                else:
                    print()
                    
                num_params += i
    print('[*] Parameter Size: {}'.format(num_params))
    print("=" * 98)

    return num_params

def inspect_file(path):
    print("-" * 10)
    print("Source:", path)
    print("-" * 10)
    print(f" - File size: {os.path.getsize(path)} bytes")
    print(f" - {torchaudio.info(path)}")
    print()