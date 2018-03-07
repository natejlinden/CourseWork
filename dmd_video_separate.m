%Nathaniel Linden
%Foreground/Background Separation 
%USES THE DMD ALGORITHM
%inputVideo is a video stream with a static background and a dynamic
%foreground
%returns:foregroundFeed and backgroundFeed (video feeds with same (x,y,t)
%dimensions as the input video but containing the foreground and background
%they are also uint8 arrays
function [foregroundFeed, backgroundFeed, fullFeed] = dmd_video_separate(inputVideo)
    v1 = VideoReader(inputVideo);
    %get video stats
    vidHeight = v1.Height; time = v1.Duration; vidWidth = v1.Width;
    %down-sample dimensions
    sz1 = vidHeight/2; sz2 = vidWidth/2;
    %get frame count
    numFrms = round(v1.FrameRate*time);
    %create time vector
    dt = time/numFrms;
    t = [0:dt:time-dt];
    
    %load frames and preprocess
    VIDEO = zeros(sz1*sz2,int64(numFrms),'uint8');
    iter = 1;
    while hasFrame(v1)
            %load
            frame = readFrame(v1);
            %convert to grayscale
            frmBW = 0.299*frame(:,:,1) + 0.587*frame(:,:,2) + 0.115*frame(:,:,3);
            %subsample - decrease by 1/2
            frmBW = imresize(frmBW,[sz1,sz2]);
            VIDEO(:,iter) = frmBW(:);
        iter = iter+1;
    end

    %create data matrices
    X = double(VIDEO);
    X1 = X(:,1:end-1);
    X2 = X(:,2:end);

    %take the SVD
    [U, S, V] = svd(X1,'econ');
    
    %show user singular value spectra and get input for r
%     figure(1)
%     plot(diag(S)/sum(diag(S)),'o')
%     hold on
%     r = input('How many modes should be used?');
r = 4;
    
    % DMD BODY
    
    %low rank truncate
    Ur = U(:,1:r);
    Sr = S(1:r, 1:r);
    Vr = V(:,1:r);

    %get Atilde
    Atilde = Ur' * X2 * Vr/Sr;

    %get eigen space of Atilde
    [W D] = eig(Atilde);
    Phi = X2 * Vr/Sr * W;
    lambda = diag(D);
    omega = log(lambda);

    %regress to b using psuedo inv
    x_1 = X(:,1); %x_1 is first frame
    b = pinv(Phi)*x_1;
    
    %get time dynamics
    time_dyn = zeros(r,length(t));
    for i = 1:length(t)
        time_dyn(:,i) = b .* exp(omega*t(iter));
    end
    %reconstruct DMD modes
    Xdmd_lr = Phi*time_dyn;
    
    % ***foreground background separation step***
    Xsparse1 = X - abs(Xdmd_lr); %abs to capture only real parts

    %use R to remove any negative pixel values
    R = Xsparse1;
    R(R > 0) = 0; 

    %construct foreground and background
    XlowRank = (R + abs(Xdmd_lr)); %Xlowrank should capture background
    Xsparse = Xsparse1 - R; %Xspase should capture foreground
    
    %back to videos
    vidSparse = zeros(sz1,sz2,round(numFrms-1));
    vidLowRank = zeros(sz1,sz2,round(numFrms));
    vidFull = zeros(sz1,sz2,round(numFrms));
    for i = 1:numFrms-1
        vidSparse(:,:,i) = reshape(Xsparse(:,i),[sz1,sz2]);
        vidLowRank(:,:,i) = reshape(XlowRank(:,i),[sz1,sz2]);
        vidFull(:,:,i) = reshape(X(:,i),[sz1,sz2]);
    end
    
    backgroundFeed = uint8(vidLowRank);
    foregroundFeed = uint8(vidSparse);
    fullFeed = uint8(vidFull);
end