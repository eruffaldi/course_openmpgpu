#include "imagex.hpp"

int main(int argc, char const *argv[])
{

	if(argc > 1)
	{
		auto src = imread<uint8_t>(argv[1]);
		std::cout << "Returned " << src.width() << " " << src.height() << std::endl;
		if(argc > 2) 
			imwrite(src,argv[2],3);
	}

	Image<int> test(4,3);
	std::cout << "assign " << test.width() << " " << test.height() << " stride " << test.stride() << std::endl;
	for(int j = 0; j < test.height(); j++)
		for(int i = 0; i < test.width(); i++)
			test(i,j) = j*10+i;
	std::cout << "inited flat:";
	for(int i = 0; i < test.numel(); i++)
	std::cout << test(i) << " ";
	std::cout << std::endl;	

	std::cout << "byrow:";
	for(auto pit = test.byrow_iterator_pair(); pit.first != pit.second; ++pit.first)
		std::cout << *pit.first << ' ';
	std::cout << std::endl;
	std::cout << "bycol:";
	for(auto pit = test.bycol_iterator_pair(); pit.first != pit.second; ++pit.first)
		std::cout << *pit.first << ' ';

	std::cout << std::endl;

	std::cout << "byrow_rev:" << std::flush;
	for(auto pit = test.byrow_rev_iterator_pair(); pit.first != pit.second; ++pit.first)
		std::cout << *pit.first << ' ';
	std::cout << std::endl;

	std::cout << "bycol_rev:" << std::flush;
	for(auto pit = test.bycol_rev_iterator_pair(); pit.first != pit.second; ++pit.first)
		std::cout << *pit.first << ' ';
	std::cout << std::endl;

	test = {1,2,3,4,   5,6,7,8,   9,10,11,12};
	std::cout << "after assign by row we have: byrow:";
	for(auto pit = test.byrow_iterator_pair(); pit.first != pit.second; ++pit.first)
		std::cout << *pit.first << ' ';
	std::cout << std::endl;

	std::cout << "inited {}:";
	for(int i = 0; i < test.numel(); i++)
	std::cout << test(i) << " ";
	std::cout << std::endl;	


	return 0;
}