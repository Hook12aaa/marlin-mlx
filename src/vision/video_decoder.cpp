#include "vision/video_decoder.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

#include <algorithm>
#include <cmath>
#include <vector>

namespace marlin {

namespace mx = mlx::core;

Result<DecodedFrames> decode_video(
    const std::string& path,
    float target_fps,
    int min_frames,
    int max_frames,
    int max_pixels) {

    AVFormatContext* fmt_ctx = nullptr;
    if (avformat_open_input(&fmt_ctx, path.c_str(), nullptr, nullptr) < 0) {
        return std::unexpected(Error::from("Failed to open video: " + path));
    }

    if (avformat_find_stream_info(fmt_ctx, nullptr) < 0) {
        avformat_close_input(&fmt_ctx);
        return std::unexpected(Error::from("Failed to find stream info"));
    }

    int video_idx = -1;
    for (unsigned i = 0; i < fmt_ctx->nb_streams; ++i) {
        if (fmt_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            video_idx = static_cast<int>(i);
            break;
        }
    }
    if (video_idx < 0) {
        avformat_close_input(&fmt_ctx);
        return std::unexpected(Error::from("No video stream found"));
    }

    auto* codecpar = fmt_ctx->streams[video_idx]->codecpar;
    auto* codec = avcodec_find_decoder(codecpar->codec_id);
    if (!codec) {
        avformat_close_input(&fmt_ctx);
        return std::unexpected(Error::from("Unsupported codec"));
    }

    auto* codec_ctx = avcodec_alloc_context3(codec);
    avcodec_parameters_to_context(codec_ctx, codecpar);
    if (avcodec_open2(codec_ctx, codec, nullptr) < 0) {
        avcodec_free_context(&codec_ctx);
        avformat_close_input(&fmt_ctx);
        return std::unexpected(Error::from("Failed to open codec"));
    }

    double duration_s = static_cast<double>(fmt_ctx->duration) / AV_TIME_BASE;
    int total_frames_target = std::clamp(
        static_cast<int>(duration_s * target_fps),
        min_frames, max_frames);

    int src_w = codec_ctx->width;
    int src_h = codec_ctx->height;

    int align = 32;
    int dst_h = (src_h / align) * align;
    int dst_w = (src_w / align) * align;
    if (dst_h < align) dst_h = align;
    if (dst_w < align) dst_w = align;

    auto* sws_ctx = sws_getContext(
        src_w, src_h, codec_ctx->pix_fmt,
        dst_w, dst_h, AV_PIX_FMT_RGB24,
        SWS_BICUBIC, nullptr, nullptr, nullptr);
    if (!sws_ctx) {
        avcodec_free_context(&codec_ctx);
        avformat_close_input(&fmt_ctx);
        return std::unexpected(Error::from("Failed to create swscale context"));
    }

    AVFrame* frame = av_frame_alloc();
    AVFrame* rgb_frame = av_frame_alloc();
    rgb_frame->format = AV_PIX_FMT_RGB24;
    rgb_frame->width = dst_w;
    rgb_frame->height = dst_h;
    av_image_alloc(rgb_frame->data, rgb_frame->linesize,
                   dst_w, dst_h, AV_PIX_FMT_RGB24, 1);

    auto* stream = fmt_ctx->streams[video_idx];
    auto tb = stream->time_base;

    std::vector<float> all_pixels;
    int decoded_count = 0;

    for (int i = 0; i < total_frames_target; ++i) {
        double target_ts = i / static_cast<double>(target_fps);
        int64_t seek_ts = static_cast<int64_t>(target_ts / av_q2d(tb));

        av_seek_frame(fmt_ctx, video_idx, seek_ts, AVSEEK_FLAG_BACKWARD);
        avcodec_flush_buffers(codec_ctx);

        bool got_frame = false;
        AVPacket pkt;
        while (av_read_frame(fmt_ctx, &pkt) >= 0) {
            if (pkt.stream_index != video_idx) {
                av_packet_unref(&pkt);
                continue;
            }
            avcodec_send_packet(codec_ctx, &pkt);
            while (avcodec_receive_frame(codec_ctx, frame) == 0) {
                double frame_ts = frame->pts * av_q2d(tb);
                if (frame_ts >= target_ts - 0.001) {
                    sws_scale(sws_ctx,
                              frame->data, frame->linesize, 0, src_h,
                              rgb_frame->data, rgb_frame->linesize);

                    for (int y = 0; y < dst_h; ++y) {
                        for (int x = 0; x < dst_w; ++x) {
                            uint8_t* p = rgb_frame->data[0] + y * rgb_frame->linesize[0] + x * 3;
                            all_pixels.push_back((p[0] / 255.0f - 0.5f) / 0.5f);
                            all_pixels.push_back((p[1] / 255.0f - 0.5f) / 0.5f);
                            all_pixels.push_back((p[2] / 255.0f - 0.5f) / 0.5f);
                        }
                    }
                    ++decoded_count;
                    got_frame = true;
                    break;
                }
            }
            av_packet_unref(&pkt);
            if (got_frame) break;
        }
    }

    av_freep(&rgb_frame->data[0]);
    av_frame_free(&rgb_frame);
    av_frame_free(&frame);
    sws_freeContext(sws_ctx);
    avcodec_free_context(&codec_ctx);
    avformat_close_input(&fmt_ctx);

    if (decoded_count == 0) {
        return std::unexpected(Error::from("No frames decoded from " + path));
    }

    auto pixels = mx::array(
        all_pixels.data(),
        {decoded_count, dst_h, dst_w, 3},
        mx::float32);

    return DecodedFrames{
        .pixels = mx::astype(pixels, mx::bfloat16),
        .num_frames = decoded_count,
        .width = dst_w,
        .height = dst_h,
        .fps_sampled = target_fps,
    };
}

}  // namespace marlin
