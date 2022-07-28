#if 1
#include <Kokkos_Core.hpp>
#include <map>
#include <chrono>

void MGT_breakpoint() {
}

double initialTime = 0.0;

std::string sFOR ("parallel for");
std::string sREDUCE ("parallel reduce");
std::string sSCAN ("parallel scan");

std::string sUNDEF ("UNDEF");

struct TimeInfo {
  std::string* mType;
  std::string *mName;
  Kokkos::Timer* mTimer;
  double mTotal;
  unsigned int mCount;
};


std::map<std::string, struct TimeInfo*> Timers;
//std::map<uint64_t*, struct TimeInfo*> Timers;

struct TimeInfo* CurrentTimer=nullptr;

void resetTimer(const char*Name, const uint32_t A, uint64_t* B)
{
    auto timer = Timers.find(Name);
    //auto timer = Timers.find(B);

//std::cout << "Parameters of resetTimer " << Name << " " << A << " " << B << std::endl;
    if (timer != Timers.end())  {
	    CurrentTimer = timer->second;
	    CurrentTimer->mTimer->reset();
//std::cout << "Found " << Name << " " << *CurrentTimer->mName << " " << A << " " << B << std::endl;
    } else {
//std::cout << "Name " << Name << " " << A << " " << B << std::endl;

//if (Name) std::cout << "NOT Found " << Name << " " << A << " " << B << std::endl;
//else std::cout << "NOT Found " << sUNDEF << " " << CurrentTimer->mName << " " << A << " " << B << std::endl;

      Kokkos::Timer* kokkACCTimer = new Kokkos::Timer;
      kokkACCTimer->reset();
      CurrentTimer = new struct TimeInfo;
      CurrentTimer->mTimer = kokkACCTimer;
      CurrentTimer->mTotal = 0.0;
      CurrentTimer->mCount = 0;
      CurrentTimer->mType = &sUNDEF;
      if (Name)
        CurrentTimer->mName = new std::string(Name);
      else
        CurrentTimer->mName = &sUNDEF;

//std::cout << "Insert " << Name << " " << A << " " << B << std::endl;
      //Timers.insert(timer, {B, CurrentTimer});
      Timers.insert(timer, {Name, CurrentTimer});
    }  
}

void readTimer_pf(uint64_t B)
{

    auto now = std::chrono::system_clock::now();
    auto now_ms = std::chrono::time_point_cast<std::chrono::microseconds>(now);
    auto epoch = now_ms.time_since_epoch();
    auto value = std::chrono::duration_cast<std::chrono::microseconds>(epoch);
    double timeStamp = value.count() - initialTime;;

    CurrentTimer->mType = &sFOR;
    double sample = CurrentTimer->mTimer->seconds();
    CurrentTimer->mCount++; 
    CurrentTimer->mTotal += sample;
    //double total = CurrentTimer->mTotal;
    double average = (CurrentTimer->mTotal/(double)CurrentTimer->mCount);
    CurrentTimer->mTimer->reset();
    
    //std::cerr << timeStamp << ",parallel_for," << *CurrentTimer->mName <<  ",Sample," << sample << ",Total," << CurrentTimer->mTotal << ",Average," << average << ",Count," << CurrentTimer->mCount << std::endl;

}

void readTimer_pr(uint64_t B)
{
    auto now = std::chrono::system_clock::now();
    auto now_ms = std::chrono::time_point_cast<std::chrono::microseconds>(now);
    auto epoch = now_ms.time_since_epoch();
    auto value = std::chrono::duration_cast<std::chrono::microseconds>(epoch);
    double timeStamp = value.count() - initialTime;;

    CurrentTimer->mType = &sREDUCE;
    double sample = CurrentTimer->mTimer->seconds();
    CurrentTimer->mCount++;
    CurrentTimer->mTotal += sample;
    //double total = CurrentTimer->mTotal;
    double average = (CurrentTimer->mTotal/(double)CurrentTimer->mCount);
    CurrentTimer->mTimer->reset();

    //std::cerr << timeStamp << ",parallel_reduce," << *CurrentTimer->mName <<  ",Sample," << sample << ",Total," << CurrentTimer->mTotal << ",Average," << average << ",Count," << CurrentTimer->mCount << std::endl;

}

void readTimer_ps(uint64_t B)
{
    auto now = std::chrono::system_clock::now();
    auto now_ms = std::chrono::time_point_cast<std::chrono::microseconds>(now);
    auto epoch = now_ms.time_since_epoch();
    auto value = std::chrono::duration_cast<std::chrono::microseconds>(epoch);
    double timeStamp = value.count() - initialTime;;

    CurrentTimer->mType = &sSCAN;
    double sample = CurrentTimer->mTimer->seconds();
    CurrentTimer->mCount++;
    CurrentTimer->mTotal += sample;
    //double total = CurrentTimer->mTotal;
    double average = (CurrentTimer->mTotal/(double)CurrentTimer->mCount);
    CurrentTimer->mTimer->reset();

    //std::cerr << timeStamp << ",parallel_scan," << *CurrentTimer->mName <<  ",Sample," << sample << ",Total," << CurrentTimer->mTotal << ",Average," << average << ",Count," << CurrentTimer->mCount << std::endl;

}


double readCurrentTimer() 
{
    auto now = std::chrono::system_clock::now();
    auto now_ms = std::chrono::time_point_cast<std::chrono::microseconds>(now);
    auto epoch = now_ms.time_since_epoch();
    auto value = std::chrono::duration_cast<std::chrono::microseconds>(epoch);
    double timeStamp = value.count();
    return (timeStamp-initialTime);
}

void initTimeProbes() 
{
  auto now = std::chrono::system_clock::now();
  auto now_ms = std::chrono::time_point_cast<std::chrono::microseconds>(now);
  auto epoch = now_ms.time_since_epoch();
  auto value = std::chrono::duration_cast<std::chrono::microseconds>(epoch);
  initialTime = value.count();

  Kokkos::Tools::Experimental::set_begin_parallel_for_callback(resetTimer);
  Kokkos::Tools::Experimental::set_end_parallel_for_callback(readTimer_pf);
  Kokkos::Tools::Experimental::set_begin_parallel_reduce_callback(resetTimer);
  Kokkos::Tools::Experimental::set_end_parallel_reduce_callback(readTimer_pr);
  Kokkos::Tools::Experimental::set_begin_parallel_scan_callback(resetTimer);
  Kokkos::Tools::Experimental::set_end_parallel_scan_callback(readTimer_ps);
}

void dumpTimers()
{
  std::cout << "+++ Time: -----------------------   Dumping timers ...\n";
  for (auto x: Timers) {
    std::cout << "+++ Time: ----, " << *x.second->mType << "," << *x.second->mName << "," << x.second->mTotal << "," << x.second->mCount << '\n';
  }
  std::cout << "+++ Time: -----------------------   End dumping timers ...\n";
}

#endif

