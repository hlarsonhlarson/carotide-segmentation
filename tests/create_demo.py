"""Generate an explicitly synthetic CT cylinder for development; no patient data."""
import sys
from pathlib import Path
import numpy as np
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, CTImageStorage, generate_uid

def create(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    study, series, frame = generate_uid(), generate_uid(), generate_uid()
    yy, xx = np.indices((32,32))
    pixels = np.where((yy-16)**2+(xx-16)**2 <= 9, 350, 0).astype(np.int16)
    for z in range(40):
        meta=FileMetaDataset()
        meta.MediaStorageSOPClassUID=CTImageStorage
        meta.MediaStorageSOPInstanceUID=generate_uid()
        meta.TransferSyntaxUID=ExplicitVRLittleEndian
        ds=FileDataset(str(folder/f'{z:03}.dcm'),{},file_meta=meta,preamble=b'\0'*128)
        ds.SOPClassUID=meta.MediaStorageSOPClassUID; ds.SOPInstanceUID=meta.MediaStorageSOPInstanceUID
        ds.PatientName='SYNTHETIC^DEMO';ds.PatientID='SYNTHETIC';ds.StudyInstanceUID=study;ds.SeriesInstanceUID=series;ds.FrameOfReferenceUID=frame
        ds.SeriesDescription='Synthetic cylinder — test data';ds.Modality='CT';ds.InstanceNumber=z+1
        ds.ImagePositionPatient=[0.,0.,float(z)];ds.ImageOrientationPatient=[1.,0.,0.,0.,1.,0.]
        ds.PixelSpacing=[1.,1.];ds.SliceThickness=1.;ds.Rows=32;ds.Columns=32
        ds.SamplesPerPixel=1;ds.PhotometricInterpretation='MONOCHROME2';ds.BitsAllocated=16;ds.BitsStored=16;ds.HighBit=15;ds.PixelRepresentation=1
        ds.RescaleIntercept=0.;ds.RescaleSlope=1.;ds.PixelData=pixels.tobytes()
        ds.save_as(ds.filename,enforce_file_format=True)
if __name__=='__main__':create(sys.argv[1])
