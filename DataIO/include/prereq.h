#if defined _WIN32 || defined __CYGWIN__
  #ifdef DATAIO_EXPORT
    #ifdef __GNUC__
      #define DATAIO_PUBLIC __attribute__ ((dllexport))
    #else
      #define DATAIO_PUBLIC __declspec(dllexport) // Note: actually gcc seems to also supports this syntax.
    #endif
  #else
    #ifdef __GNUC__
      #define DATAIO_PUBLIC __attribute__ ((dllimport))
    #else
      #define DATAIO_PUBLIC __declspec(dllimport) // Note: actually gcc seems to also supports this syntax.
    #endif
  #endif
  #define DATAIO_LOCAL
#else
  #if __GNUC__ >= 4
    #define DATAIO_PUBLIC __attribute__ ((visibility ("default")))
    #define DATAIO_LOCAL  __attribute__ ((visibility ("hidden")))
  #else
    #define DATAIO_PUBLIC
    #define DATAIO_LOCAL
  #endif
#endif