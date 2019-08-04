import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys
import warnings
warnings.filterwarnings('ignore')

#----------------------------------
# matplotlibのデフォルトの設定
#----------------------------------
# フォントサイズ
plt.rcParams["font.size"] = 12
# 線幅の変更
plt.rcParams['lines.linewidth'] = 2
# xの範囲のマージンをなくす
plt.rcParams['axes.xmargin'] = 0
#画像の限界幅
maxWidth = 256

# 標準化
def min_max(x, axis=None):
    x_min = x.min(axis=axis, keepdims=True)
    x_max = x.max(axis=axis, keepdims=True)
    return (x - x_min) / (x_max - x_min)

# ガウス分布
def norm2d(x,y,sigma):
    Z = np.exp(-(x**2 + y**2) / (2 * sigma**2)) / (2 * np.pi * sigma**2)
    return Z

# ガウスフィルター
def gaussian_kernel(size):
    # サイズの確認
    if size%2==0:
        return
    sigma = (size-1)/2

    # [0,size]→[-sigma, sigma] にずらす
    x = y = np.arange(0,size) - sigma
    X,Y = np.meshgrid(x,y)
    mat = norm2d(X,Y,sigma)

    # 総和が1になるように標準化
    kernel = mat / np.sum(mat)
    kernel = min_max(kernel)
    return kernel

# ガウスマスク
def gauss_mask(x,y,size):
    a1 = np.zeros((rows,cols,2))

    # 平均化マスク
    if size <= 1:
        if ms == 1:
            a1[y,x] = 1
        else:
            y1 = y-ms
            y2 = y+ms
            x1 = x-ms
            x2 = x+ms
            if y1<0 :
                y1 = 0
            if y2>rows:
                y2 = rows
            if x1<0:
                x1 = 0
            if x2>cols:
                x2 =cols
            a1[y1:y2,x1:x2] = 1
        return a1

    # ガウスマスク
    b1 = np.zeros((size,size,2))
    s = int((size-1)/2)
    b = gaussian_kernel(size)
    b1[:,:,0] = b
    b1[:,:,1] = b
    gy1 = y-s
    gy2 = y+s+1
    gx1 = x-s
    gx2 = x+s+1
    if gy1<0 :
        gy1 = 0
    if gy2>rows:
        gy2 = rows
    if gx1<0:
        gx1 = 0
    if gx2>cols:
        gx2 =cols
    a1[gy1:gy2,gx1:gx2] = b1[s-(gy1+y):s+(gy2-y),s-(gx1+x):s+(gx2-x)]
    return a1

# マウスを押した時
def Press(event):
    global DragFlag, Times, mask, z, ms

    # Noneなら終了
    if (event.xdata is None) or (event.xdata is None) or (event.x <= 300):
        return

    # 更新
    if (event.x <= 600) and (event.y <= 300):
        # xとyの座標
        x = int(event.xdata+0.5)
        y = int(event.ydata+0.5)

        # マスクの更新
        mask += gauss_mask(x,y,z)
        black = gauss_mask(x,y,z)
        mask = np.where(mask > 1, 1, mask)
        mask_img = mask[:,:,0]

        # フーリエ変換
        fshift = dft_shift*mask
        f_ishift = np.fft.ifftshift(fshift)
        img_back = cv2.idft(f_ishift)
        img_back = img_back[:,:,0]
        gshift = dft_shift*black
        g_ishift = np.fft.ifftshift(gshift)
        g_back = cv2.idft(g_ishift)
        g_back = g_back[:,:,0]

        # 右クリック
        if event.button==3:
            if(Times==0):
                DragFlag = True
                Times = 1
            else:
                DragFlag = False
                Times = 0

    # リセットボタン
    elif (event.x > 600) and (event.y > 300):
        # 初期化
        mask = np.zeros((rows,cols,2))
        mask_img = mask[:,:,0]
        DragFlag = False
        Times = 0
        z = 1
        ms = 1
        img_back = np.zeros(img.shape)
        g_back = np.zeros(img.shape)

    # Plotの更新
    plt.subplot(234),plt.cla(),plt.imshow(img_back, cmap = 'gray')
    plt.title('IFFTed Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(235),plt.cla(),plt.imshow(mask_img, cmap = 'gray')
    plt.title('Unmasked Amplitude Spectrum'),plt.xticks([]), plt.yticks([])
    plt.subplot(236),plt.cla(),plt.imshow(g_back, cmap = 'gray')
    plt.title('Recent Grating Added'), plt.xticks([]), plt.yticks([])
    plt.draw()


# マウスをドラッグした時
def Drag(event):
    global DragFlag, Times, mask, z

    # ドラッグしていなければ終了
    if DragFlag == False:
        return

    # Noneなら終了
    if (event.xdata is None) or (event.ydata is None):
        DragFlag = False
        Times = 0
        return

    # xとyの座標
    x = int(event.xdata+0.5)
    y = int(event.ydata+0.5)

    # マスクの更新
    mask += gauss_mask(x,y,z)
    black = gauss_mask(x,y,z)
    mask = np.where(mask > 1, 1, mask)
    mask_img = mask[:,:,0]

    # フーリエ変換
    fshift = dft_shift*mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = img_back[:,:,0]
    gshift = dft_shift*black
    g_ishift = np.fft.ifftshift(gshift)
    g_back = cv2.idft(g_ishift)
    g_back = g_back[:,:,0]

    # Plotの更新
    plt.subplot(234),plt.cla(),plt.imshow(img_back, cmap = 'gray')
    plt.title('IFFTed Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(235),plt.cla(),plt.imshow(mask_img, cmap = 'gray')
    plt.title('Unmasked Amplitude Spectrum'),plt.xticks([]), plt.yticks([])
    plt.subplot(236),plt.cla(),plt.imshow(g_back, cmap = 'gray')
    plt.title('Recent Grating Added'), plt.xticks([]), plt.yticks([])
    plt.draw()

# キーボードを押した時
def onKey(event):
    global z, ms

    # 平均化マスクのサイズ変更
    if event.key == 'up':
        z = 1
        if ms > max:
            ms = max
        else:
            ms += 1
    elif event.key == 'down':
        z = 1
        if ms <= 1:
            ms = 1
        else:
            ms -= 1

    # ガウスマスクのサイズ変更
    elif event.key == 'right':
        if z > max:
            z = max
        else:
            z += 2
    elif event.key == 'left':
        if z <= 1:
            z = 1
        else:
            z -= 2

    # 'q'で終了
    elif event.key == 'q':
        plt.close(event.canvas.figure)
    sys.stdout.flush()

#----------------------------------
# 画像
#----------------------------------
fnm = 'imgs/Lenna.jpg'
originalImage = cv2.imread(fnm,0)
(oh,ow) = originalImage.shape
if ow>maxWidth:
    img = cv2.resize(originalImage,(maxWidth,int(oh*maxWidth/ow)),interpolation=cv2.INTER_AREA)
else:
    img = originalImage
reset = cv2.imread('imgs/button_reset.jpg')

#----------------------------------
# フーリエ変換
#----------------------------------
# フーリエ変換
rows, cols = img.shape
dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
amplitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))

# マスクの初期設定
black = np.zeros((rows,cols,2))
mask = np.zeros((rows,cols,2))
mask_img = mask[:,:,0]

# 逆フーリエ変換
fshift = dft_shift*mask
f_ishift = np.fft.ifftshift(fshift)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])


#----------------------------------
# 初期化
#----------------------------------
#カーソルの最大値
max = int(min(rows,cols)/4)
if max%2 == 0:
    max += 1

#カーソルの大きさ
z = 1
ms = 1

# ドラッグしているかのフラグ
DragFlag = False
Times = 0

#----------------------------------
# Plot
#----------------------------------
fig = plt.figure(figsize=(9,6))
plt.subplot(231),plt.imshow(img, cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(232),plt.imshow(amplitude_spectrum, cmap = 'gray')
plt.title('Amplitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.subplot(233),plt.imshow(reset)
plt.xticks([]), plt.yticks([])
plt.subplot(234),plt.imshow(img_back, cmap = 'gray')
plt.title('IFFTed Image'), plt.xticks([]), plt.yticks([])
plt.subplot(235),plt.imshow(mask_img, cmap = 'gray')
plt.title('Unmasked Amplitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.subplot(236),plt.imshow(img_back, cmap = 'gray')
plt.title('Recent Grating Added'), plt.xticks([]), plt.yticks([])

#-------------------------------------------------------------------------------
# 共通の設定
#-------------------------------------------------------------------------------

# レイアウトを整える
plt.tight_layout()

# イベントの登録
plt.connect('button_press_event', Press)     # マウスを押した時
plt.connect('motion_notify_event', Drag)     # マウスをドラッグ(右クリック)した時
plt.connect('key_press_event',onKey)         # キーボードを押した時

# 描画
plt.show()
