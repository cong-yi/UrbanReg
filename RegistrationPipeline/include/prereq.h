#if defined _WIN32 || defined __CYGWIN__
  #ifdef REGPIPELINE_EXPORT
    #ifdef __GNUC__
      #define REGPIPELINE_PUBLIC __attribute__ ((dllexport))
    #else
      #define REGPIPELINE_PUBLIC __declspec(dllexport) // Note: actually gcc seems to also supports this syntax.
    #endif
  #else
    #ifdef __GNUC__
      #define REGPIPELINE_PUBLIC __attribute__ ((dllimport))
    #else
      #define REGPIPELINE_PUBLIC __declspec(dllimport) // Note: actually gcc seems to also supports this syntax.
    #endif
  #endif
  #define REGPIPELINE_LOCAL
#else
  #if __GNUC__ >= 4
    #define REGPIPELINE_PUBLIC __attribute__ ((visibility ("default")))
    #define REGPIPELINE_LOCAL  __attribute__ ((visibility ("hidden")))
  #else
    #define REGPIPELINE_PUBLIC
    #define REGPIPELINE_LOCAL
  #endif
#endif