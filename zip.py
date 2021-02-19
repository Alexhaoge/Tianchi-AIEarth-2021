import os
import zipfile

def compress(res_dir='result', output_dir='result.zip'):
    z = zipfile.ZipFile(output_dir, 'w')
    for d in os.listdir(res_dir):
        z.write(res_dir + os.sep + d)
    z.close()


if __name__=='__main__':
    compress()
