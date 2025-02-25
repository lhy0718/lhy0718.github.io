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
})
