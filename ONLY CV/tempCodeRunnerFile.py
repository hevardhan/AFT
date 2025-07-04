 while True:
        screen = grab_screen(region=(0,40,800,640))
        print('Frame took {} seconds'.format(time.time()-last_time))
        last_time = time.time()
        new_screen,original_image, m1, m2 = process_img(screen)
        #cv2.imshow('window', new_screen)
        cv2.imshow('window2',cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))

        if m1 < 0 and m2 < 0:
            right()
        elif m1 > 0  and m2 > 0:
            left()
        else:
            straight()

        #cv2.imshow('window',cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break