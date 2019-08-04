# Real time Fast Fourier Transform
====

# Overview

  マウスで周波数を指定し、フーリエ変換・逆変換をリアルタイムに行う[デモ](https://www.youtube.com/watch?v=qB0cffZpw-A)を実装した。

## Description

  OpenCVで画像を取得し，マウスで周波数を指定し(周波数フィルタリング)、フーリエ変換・逆変換をリアルタイムに行う。

## Demo

  ![](https://github.com/Black-Field/RFFT/blob/master/misk/demo.gif)

## Requirement

  Pythonのバージョンと使用したライブラリを以下に示す。
  > Python : 3.7.3
  >> opencv-python==4.1.0.25  
  >> numpy==1.16.4  
  >> matplotlib==3.1.1  
  
  MacBook Air (13-inch, Early 2015)で動作確認を行った。機能面、特に速度はいまいちであるがデモとしては十分な性能を持つ。OpenCVのdftのライブラリーを使用したが、連続モード時、マウスの速度についていけないので、ゆっくりとマウスを動かして上げる必要がある。

## Usage

  1. コンソールからこのプログラム(RFFT.py)を実行する。
  2. 下の段の中央(Unmasked Amplitude Spectrum)で上の(Amplitude Spectrum)に対応する場所をクリックすると対応する正弦波が右(Recent Grating Added)に表示され、左(IFFTed Image)にこれまでに追加された正弦波の和が表示される。
  3. 'q' を押すと終了する。
  
  - 特殊機能  
   > - 右クリックで連続入力モードになり、マウスの動きに合わせて対応する正弦波が追加されていく。
   > - キーボードの'up' を押すと平均化フィルターとして周辺の対応する正弦波を追加していく。押すたびに領域が拡大する。'down' 縮小する。
   > - キーボードの'right' を押すとガウスフィルターとして周辺の対応する正弦波を追加していく。押すたびに領域が拡大する。'left' 縮小する。

## Install

  このRepositorieをダウンロードして、コンソールからこのプログラム(RFFT.py)を実行してください。

## References

  [OpenCV](http://opencv.jp/opencv-2svn/cpp/index.html)
  > OpenCVのライブラリーについて参考にした。
  
  [OpenCV 2](http://lang.sist.chukyo-u.ac.jp/classes/OpenCV/py_tutorials/py_imgproc/py_transforms/py_fourier_transform/py_fourier_transform.html)
  > フーリエ変換のやり方について参考にした。
  
  [Qiita](https://qiita.com/HajimeKawahara/items/abc24fa2216009523656)
  > matplotlibを使った各イベント情報の取得について参考にした。
