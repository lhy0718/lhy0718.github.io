document.addEventListener("DOMContentLoaded", function () {
  article_imgs = $(".page__content img")
  article_imgs.each(function () {
    $(this).on("click", function () {
      width_percent = ($(this).width() / $(this).parent().width()) * 100
      if (width_percent === 100) {
        $(this).css("width", "50%")
      } else {
        $(this).css("width", "100%")
      }
    })
  })

  url = window.location.href
  if (url.includes("posts") || url.includes("categories") || url.includes("tags")) {
    $(".taxonomy__section").hide()
  }
  for (i = 0; i < $(".taxonomy__index li").length; i++) {
     $(".taxonomy__index li")[i].onclick = function () {
      year = $(this).find("a").attr("href").split("#")[1]
      $(".taxonomy__section").hide()
      $(".taxonomy__section").each(function () {
        if ($(this).attr("id") === year) {
          $(this).show()
        }
      })
     }
  }
})