#include <hc.hpp>

#include <cassert>
#include <chrono>
#include <vector>
#include <iostream>

int main(int argc, char *argv[])
{
    std::size_t arr_size = 16 * 1024 * 1024;
    if( argc > 1 )
        arr_size = std::stol(argv[1]);

    const std::size_t half_len = arr_size / 2;
    const std::size_t bytes  = arr_size * sizeof(int);
    const float gbytes = bytes/(1024*1024*1024.);

    std::cout << "using " << arr_size  << " elements ("<< gbytes <<" GB)\n";

    std::vector<int> h_payload(arr_size, 42);
    std::vector<int> h_results(arr_size, 0 );

    std::vector<hc::completion_future> streams(2);
    std::vector<hc::array<int,1> >   on_device = {hc::array<int,1>(half_len),
                                                  hc::array<int,1>(half_len)};

    std::vector<hc::array_view<int,1> >   device_view = {hc::array_view<int,1>(on_device[0]),
                                                       hc::array_view<int,1>(on_device[1])};

    auto inbegin = h_payload.begin();
    auto outbegin = h_results.begin();

    auto start = std::chrono::high_resolution_clock::now();
    for(int i = 0;i<streams.size();i++){

        streams[i] = hc::copy_async(inbegin,inbegin+half_len,on_device[i]);
        //streams[i].wait();

        auto current_view = on_device[i].view_as(on_device[i].get_extent());
        auto future = hc::parallel_for_each(on_device[i].get_extent(),
                                            [=](hc::index<1> idx) [[hc]] {
                                                current_view[idx] *= 4;
                                            });
        //
        streams[i].then([&](){
                future.wait();
            });
        streams[i].wait();

        streams[i] = hc::copy_async(on_device[i],outbegin);
        streams[i].wait();

        inbegin += half_len;
        outbegin += half_len;
    }
    auto end = std::chrono::high_resolution_clock::now();


    if(h_results[0] == 4*h_payload[0])
        return 0;
    else{
        std::cerr << "uuups, output ("<< h_results[0] <<") is not as expected ("<< 4*h_payload[0] << ")\n";
        return 1;
    }

}
