R = stereoParams.RotationOfCamera2'
T = stereoParams.TranslationOfCamera2'
F = stereoParams.FundamentalMatrix'
E = stereoParams.EssentialMatrix'
cameraMatrix1 = stereoParams.CameraParameters1.IntrinsicMatrix'
distCoeffs1 = stereoParams.CameraParameters1.RadialDistortion;
errores1 = stereoParams.CameraParameters1.ReprojectionErrors;

cameraMatrix2 = stereoParams.CameraParameters2.IntrinsicMatrix'
distCoeffs2 = stereoParams.CameraParameters2.RadialDistortion;
errores2 = stereoParams.CameraParameters2.ReprojectionErrors;
tamano_cuadro = 12;%mm
save('/home/carlos/Documentos/Videos/calibraciones_paralelo/grab2/calib/calibData.mat',...
     'cameraMatrix1', 'distCoeffs1', 'errores1',...
     'cameraMatrix2', 'distCoeffs2', 'errores2',...
     'R', 'T', 'E', 'F', 'tamano_cuadro')