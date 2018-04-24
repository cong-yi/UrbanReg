#if defined _WIN32 || defined __CYGWIN__
  #ifdef GPS_EXPORT
    #ifdef __GNUC__
      #define GPS_PUBLIC __attribute__ ((dllexport))
    #else
      #define GPS_PUBLIC __declspec(dllexport) // Note: actually gcc seems to also supports this syntax.
    #endif
  #else
    #ifdef __GNUC__
      #define GPS_PUBLIC __attribute__ ((dllimport))
    #else
      #define GPS_PUBLIC __declspec(dllimport) // Note: actually gcc seems to also supports this syntax.
    #endif
  #endif
  #define GPS_LOCAL
#else
  #if __GNUC__ >= 4
    #define GPS_PUBLIC __attribute__ ((visibility ("default")))
    #define GPS_LOCAL  __attribute__ ((visibility ("hidden")))
  #else
    #define GPS_PUBLIC
    #define GPS_LOCAL
  #endif
#endif